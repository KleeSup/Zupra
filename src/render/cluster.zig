//! src/render/cluster.zig
//!
//! Clustered lighting — the froxel grid and the CPU light-assignment pass.
//!
//! The view frustum is divided into a 3D grid of "froxels" (frustum voxels):
//! grid.x by grid.y screen tiles, times grid.z depth slices. Each froxel gets a
//! short list of the punctual lights that reach it, so a shaded pixel iterates
//! only its own froxel's list rather than every light in the scene. That's what
//! decouples per-pixel cost from total light count — the thing a 16-light
//! uniform array can't do at any array size.
//!
//! DEPTH SLICES ARE EXPONENTIAL, not linear. With linear slices almost every
//! froxel lands in the distant half of the frustum where froxels are enormous
//! and useless, while everything near the camera crams into one slice. The
//! standard exponential mapping (Olsson/Assarsson) keeps froxels roughly
//! cube-shaped through the whole depth range:
//!
//!     slice = floor(log(z) * (N / log(far/near)) - (N * log(near) / log(far/near)))
//!
//! The shader derives its slice from view depth with that same closed form — no
//! search, no loop.
//!
//! WHY TEXTURES, NOT STORAGE BUFFERS: WebGL2/GLES 3.0 have float textures,
//! integer textures and texelFetch in core, but no SSBOs. Keeping cluster data
//! in textures means pbr_lib.glsl.inc has ONE read path on every backend, so
//! shadows (and everything after) get written once instead of twice.
//!
//! WHY FIXED CAPACITY PER CLUSTER: the textbook GPU kernel appends to a shared
//! index list using an atomic counter — but atomics don't work on storage
//! images, and WebGPU's write-only storage textures rule them out entirely.
//! Giving each cluster a fixed slot range removes the need for atomics, bounds
//! worst-case per-pixel cost (arguably a feature), and means the future compute
//! builder can write the exact same layout the CPU builder does.
//!
//! TEXTURE LAYOUTS (contract with pbr_lib.glsl.inc):
//!
//!   cluster table: RG32UI, one texel per froxel, (offset, count)
//!                  width  = grid.x
//!                  height = grid.y * grid.z   (slices stacked vertically)
//!
//!   index list:    RGBA32UI, FOUR light indices per texel. Packing four per
//!                  texel quarters the fetch count in the shader's inner loop —
//!                  cheap to do now, a layout change to retrofit later.

const std = @import("std");
const sg = @import("sokol").gfx;
const math = @import("../math.zig");
const light_mod = @import("light.zig");

const Vec3 = math.Vec3;
const Light = light_mod.Light;
const Camera3D = @import("camera3d.zig").Camera3D;

/// Froxel grid dimensions. 16x9 matches a 16:9 aspect so screen tiles stay
/// roughly square; 24 depth slices is the common default (Doom 2016 used 24).
pub const GridDims = struct {
    x: u32 = 16,
    y: u32 = 9,
    z: u32 = 24,

    pub fn total(self: GridDims) u32 {
        return self.x * self.y * self.z;
    }
};

/// Max punctual lights per froxel. 64 overlapping lights in one small volume is
/// already a lot; if content exceeds it the honest fixes are a finer grid or
/// importance-based dropping, not an unbounded list (which would mean unbounded
/// per-pixel cost).
pub const default_cluster_capacity: u32 = 64;

/// What the froxel grid actually holds after a build.
///
/// Per-pixel shading cost in a clustered renderer is (lights in this fragment's
/// froxel) x (cost per light), and the second term is fixed. So froxel occupancy
/// IS the performance model: without it, "why is this camera angle slow" can
/// only be answered by toggling features until something moves.
pub const Stats = struct {
    /// Highest occupancy of any froxel. Governs the worst case on screen, and is
    /// the number to watch when frame time swings with camera angle.
    max_lights: u32 = 0,
    /// Mean over OCCUPIED froxels only. Averaging across the whole grid buries
    /// the interesting figure under a large mostly-empty volume -- most of a
    /// frustum is empty air, and reporting that as "0.3 lights per froxel" says
    /// nothing about what the shaded pixels are paying.
    avg_lights: f32 = 0,
    /// Froxels holding at least one light.
    occupied: u32 = 0,
    /// Froxels filled to capacity. Any further light touching one of these was
    /// discarded, so fragments there are lit with an incomplete light list --
    /// visible as lights that vanish at certain angles.
    saturated: u32 = 0,
    /// Individual assignments discarded because the froxel was already full.
    /// Non-zero means the scene has outgrown cluster_capacity somewhere.
    dropped: u32 = 0,
    /// Froxels in the grid, for context.
    total: u32 = 0,
};

pub const ClusterOptions = struct {
    grid: GridDims = .{},
    capacity: u32 = default_cluster_capacity,
};

/// View-space sphere bound for a light, used for froxel intersection.
const LightBound = struct {
    center_vs: Vec3, // view-space center
    radius: f32,
};

pub const ClusterBuilder = struct {
    allocator: std.mem.Allocator,
    grid: GridDims,
    capacity: u32,

    // GPU resources
    table_tex: sg.Image = .{},
    table_view: sg.View = .{},
    index_tex: sg.Image = .{},
    index_view: sg.View = .{},

    // CPU staging
    table: std.ArrayList([2]u32) = .empty, // (offset, count) per froxel
    indices: std.ArrayList(u32) = .empty, // flat, capacity per froxel
    counts: std.ArrayList(u32) = .empty, // fill count per froxel
    bounds: std.ArrayList(LightBound) = .empty,

    /// Depth-slice constants, recomputed when the camera's near/far change.
    /// Uploaded to the shader so it derives the same slice index.
    slice_scale: f32 = 0,
    slice_bias: f32 = 0,
    near: f32 = 0,
    far: f32 = 0,

    index_tex_texels: u32 = 0,

    /// Occupancy summary from the last build. Read-only to callers.
    stats: Stats = .{},

    pub fn init(allocator: std.mem.Allocator, opts: ClusterOptions) ClusterBuilder {
        var self = ClusterBuilder{
            .allocator = allocator,
            .grid = opts.grid,
            .capacity = opts.capacity,
        };
        self.createTextures();
        return self;
    }

    pub fn deinit(self: *ClusterBuilder) void {
        self.table.deinit(self.allocator);
        self.indices.deinit(self.allocator);
        self.counts.deinit(self.allocator);
        self.bounds.deinit(self.allocator);
        self.destroyTextures();
    }

    fn destroyTextures(self: *ClusterBuilder) void {
        if (self.table_view.id != 0) sg.destroyView(self.table_view);
        if (self.table_tex.id != 0) sg.destroyImage(self.table_tex);
        if (self.index_view.id != 0) sg.destroyView(self.index_view);
        if (self.index_tex.id != 0) sg.destroyImage(self.index_tex);
        self.table_view = .{};
        self.table_tex = .{};
        self.index_view = .{};
        self.index_tex = .{};
    }

    fn createTextures(self: *ClusterBuilder) void {
        // Cluster table: one RG32UI texel per froxel. Slices stack vertically so
        // the texture stays a sane shape regardless of grid.z.
        self.table_tex = sg.makeImage(.{
            .width = @intCast(self.grid.x),
            .height = @intCast(self.grid.y * self.grid.z),
            .pixel_format = .RG32UI,
            .usage = .{ .stream_update = true },
            .label = "cluster_table",
        });
        self.table_view = sg.makeView(.{ .texture = .{ .image = self.table_tex } });

        // Index list: 4 indices per RGBA32UI texel.
        const total_indices = self.grid.total() * self.capacity;
        const texels = (total_indices + 3) / 4;
        // Wrap into rows so width stays under the max texture dimension.
        const width: u32 = 1024;
        const height: u32 = (texels + width - 1) / width;
        self.index_tex_texels = width * height;
        self.index_tex = sg.makeImage(.{
            .width = @intCast(width),
            .height = @intCast(height),
            .pixel_format = .RGBA32UI,
            .usage = .{ .stream_update = true },
            .label = "cluster_indices",
        });
        self.index_view = sg.makeView(.{ .texture = .{ .image = self.index_tex } });
    }

    /// Change the grid or capacity at runtime (mobile wants a coarser grid).
    pub fn reconfigure(self: *ClusterBuilder, opts: ClusterOptions) void {
        if (opts.grid.x == self.grid.x and opts.grid.y == self.grid.y and
            opts.grid.z == self.grid.z and opts.capacity == self.capacity) return;
        self.destroyTextures();
        self.grid = opts.grid;
        self.capacity = opts.capacity;
        self.createTextures();
    }

    /// Exponential depth-slice constants. The shader uses the same pair to get
    /// its slice from view depth, so both sides agree by construction.
    fn updateSliceParams(self: *ClusterBuilder, near: f32, far: f32) void {
        if (near == self.near and far == self.far) return;
        self.near = near;
        self.far = far;
        const zn: f32 = @floatFromInt(self.grid.z);
        const log_ratio = @log(far / near);
        self.slice_scale = zn / log_ratio;
        self.slice_bias = -(zn * @log(near) / log_ratio);
    }

    /// x = slice scale, y = slice bias, z = grid.z, w = capacity.
    /// Plus grid.x/grid.y for tile size — packed by the caller into its uniform.
    pub fn sliceParams(self: ClusterBuilder) [4]f32 {
        return .{
            self.slice_scale,
            self.slice_bias,
            @floatFromInt(self.grid.z),
            @floatFromInt(self.capacity),
        };
    }

    pub fn gridParams(self: ClusterBuilder) [4]f32 {
        return .{
            @floatFromInt(self.grid.x),
            @floatFromInt(self.grid.y),
            @floatFromInt(self.grid.z),
            0,
        };
    }

    // -----------------------------------------------------------------------
    //  Build
    // -----------------------------------------------------------------------

    /// Assign punctual lights to froxels. `lights` must be the LightStore's
    /// punctual slice — cluster indices refer to positions within it, not to the
    /// global light list.
    pub fn build(self: *ClusterBuilder, lights: []const Light, camera: Camera3D) void {
        self.updateSliceParams(camera.near, camera.far);

        const froxel_count = self.grid.total();
        self.counts.resize(self.allocator, froxel_count) catch return;
        self.table.resize(self.allocator, froxel_count) catch return;
        self.indices.resize(self.allocator, froxel_count * self.capacity) catch return;
        // Only the counts need clearing. The index list is bounded by its
        // froxel's count on every read, so stale entries beyond that count are
        // unreachable -- and clearing it means memsetting froxels x capacity
        // u32s every frame (roughly a megabyte at the default grid), which is
        // pure memory bandwidth spent to zero data nothing will look at.
        @memset(self.counts.items, 0);
        self.stats = .{};

        // Fixed slots: froxel i owns indices [i*capacity, (i+1)*capacity).
        for (self.table.items, 0..) |*e, i| {
            e.* = .{ @as(u32, @intCast(i)) * self.capacity, 0 };
        }

        if (lights.len > 0) {
            self.computeBounds(lights, camera);
            self.assign(camera);
        }

        // Publish the fill counts, and summarise them on the same pass -- the
        // data is already in cache here, so the stats are effectively free.
        var max_lights: u32 = 0;
        var occupied: u32 = 0;
        var saturated: u32 = 0;
        var sum: u64 = 0;
        for (self.table.items, 0..) |*e, i| {
            const c = self.counts.items[i];
            e.*[1] = c;
            if (c > 0) {
                occupied += 1;
                sum += c;
                max_lights = @max(max_lights, c);
                if (c >= self.capacity) saturated += 1;
            }
        }
        self.stats.max_lights = max_lights;
        self.stats.occupied = occupied;
        self.stats.saturated = saturated;
        self.stats.total = froxel_count;
        self.stats.avg_lights = if (occupied > 0)
            @as(f32, @floatFromInt(sum)) / @as(f32, @floatFromInt(occupied))
        else
            0;

        self.upload();
    }

    /// View-space bounding sphere per light. Spot lights use their cone's
    /// bounding sphere — looser than an exact cone test, but the froxel test is
    /// a broad phase and a slightly generous bound only costs a few extra
    /// entries, never correctness.
    fn computeBounds(self: *ClusterBuilder, lights: []const Light, camera: Camera3D) void {
        self.bounds.resize(self.allocator, lights.len) catch return;
        const view = camera.view();

        for (lights, 0..) |l, i| {
            var center = l.position;
            var radius = l.range;

            if (l.type == .spot) {
                // Bounding sphere of a cone: for a narrow cone (half-angle < 45deg)
                // the sphere is centered along the axis at range/(2*cos^2(half));
                // for a wide cone it's centered at the apex with the cap radius.
                const half = l.spot_outer_deg * math.DEG_TO_RAD;
                const cos_half = @cos(half);
                if (cos_half * cos_half > 0.5) {
                    const d = l.range / (2.0 * cos_half * cos_half);
                    center = l.position.add(l.direction.normalize().mul(d));
                    radius = d;
                } else {
                    radius = l.range * @sin(half);
                }
            }

            self.bounds.items[i] = .{
                .center_vs = transformPoint(view, center),
                .radius = radius,
            };
        }
    }

    /// NDC extent of a view-space sphere along one axis, by TANGENT PLANES.
    ///
    /// The naive bound -- project centre +/- radius and divide by the frustum
    /// half-extent at that depth -- treats the sphere as an axis-aligned box and
    /// is only right when it sits near the view axis. The true silhouette of a
    /// sphere seen from a point is set by the tangent lines from the eye, and it
    /// grows as the sphere moves off-axis, without bound as it approaches the
    /// eye plane. So the naive version UNDERESTIMATES exactly when a light is
    /// heading off-screen: tiles that genuinely see it are never assigned, and
    /// the result is a rectangular region of missing light that appears only at
    /// certain angles.
    ///
    /// Working in the 2D plane spanned by this axis and z: rotate the centre
    /// direction by +/- asin(r/d), where d is the distance to the centre. Those
    /// two directions are the tangent rays, and each projects to one edge of the
    /// bound.
    ///
    /// Returns null when the eye is INSIDE the sphere, where there is no
    /// silhouette and the light covers the whole screen.
    fn sphereAxisNdc(p: f32, z: f32, r: f32, tan_half: f32) ?[2]f32 {
        const d2 = p * p + z * z;
        if (d2 <= r * r) return null; // eye inside: everything

        const a = @sqrt(d2 - r * r);

        // (p, z) rotated by +/- theta, with sin(theta) = r/d, cos(theta) = a/d.
        // The 1/d scale factor cancels in the projection below, so it is omitted.
        const t1 = tangentNdc(a * p - r * z, r * p + a * z, tan_half);
        const t2 = tangentNdc(a * p + r * z, -r * p + a * z, tan_half);

        return .{ @min(t1, t2), @max(t1, t2) };
    }

    /// Project one tangent ray to NDC. A ray at or behind the eye plane has no
    /// finite projection -- the silhouette runs off that side of the screen --
    /// so it reports infinity in the direction it was heading, which the caller
    /// then clamps to the screen edge.
    fn tangentNdc(tx: f32, tz: f32, tan_half: f32) f32 {
        if (tz > 1e-5) return (tx / tz) / tan_half;
        return if (tx >= 0) std.math.inf(f32) else -std.math.inf(f32);
    }

    fn assign(self: *ClusterBuilder, camera: Camera3D) void {
        const gx = self.grid.x;
        const gy = self.grid.y;
        const gz = self.grid.z;

        // Frustum half-extents at z = 1, i.e. tan of each half-angle.
        const tan_half_y = @tan(camera.fov_y * 0.5);
        const tan_half_x = tan_half_y * camera.aspect;

        for (self.bounds.items, 0..) |b, li| {
            // View-space depth range the sphere spans (LH: +z forward).
            const z_min = @max(b.center_vs.z - b.radius, camera.near);
            const z_max = @min(b.center_vs.z + b.radius, camera.far);
            if (z_max <= camera.near or z_min >= camera.far) continue;

            // Screen bound from the true silhouette. Computed ONCE per light
            // rather than per depth slice: the tangent bound is the sphere's
            // outline as seen from the eye and does not depend on which slice is
            // being considered. That also makes this cheaper than the old
            // per-slice version it replaces.
            var x_lo: f32 = -1.0;
            var x_hi: f32 = 1.0;
            var y_lo: f32 = -1.0;
            var y_hi: f32 = 1.0;

            if (sphereAxisNdc(b.center_vs.x, b.center_vs.z, b.radius, tan_half_x)) |xr| {
                x_lo = std.math.clamp(xr[0], -1.0, 1.0);
                x_hi = std.math.clamp(xr[1], -1.0, 1.0);
                if (x_hi < -1.0 or x_lo > 1.0) continue; // fully off-screen
            }
            if (sphereAxisNdc(b.center_vs.y, b.center_vs.z, b.radius, tan_half_y)) |yr| {
                y_lo = std.math.clamp(yr[0], -1.0, 1.0);
                y_hi = std.math.clamp(yr[1], -1.0, 1.0);
                if (y_hi < -1.0 or y_lo > 1.0) continue;
            }
            // A null from either axis means the eye is inside the sphere on that
            // axis, and the full [-1, 1] default already covers it.

            const tx_min = @max(ndcToTile(x_lo, gx), 0);
            const tx_max = ndcToTile(x_hi, gx);
            const ty_min = @max(ndcToTile(y_lo, gy), 0);
            const ty_max = ndcToTile(y_hi, gy);
            if (tx_max < 0 or ty_max < 0) continue;

            const slice_min = self.sliceForDepth(z_min);
            const slice_max = self.sliceForDepth(z_max);

            var sz = slice_min;
            while (sz <= slice_max and sz < gz) : (sz += 1) {
                var ty: i32 = ty_min;
                while (ty <= ty_max and ty < @as(i32, @intCast(gy))) : (ty += 1) {
                    var tx: i32 = tx_min;
                    while (tx <= tx_max and tx < @as(i32, @intCast(gx))) : (tx += 1) {
                        const froxel = (sz * gy + @as(u32, @intCast(ty))) * gx + @as(u32, @intCast(tx));
                        const c = self.counts.items[froxel];
                        if (c >= self.capacity) {
                            // Dropped. Counted rather than silent: a light that
                            // disappears only from certain angles is otherwise
                            // one of the harder bugs to recognise, and this
                            // turns it into a number.
                            self.stats.dropped += 1;
                            continue;
                        }
                        self.indices.items[froxel * self.capacity + c] = @intCast(li);
                        self.counts.items[froxel] = c + 1;
                    }
                }
            }
        }
    }

    fn sliceForDepth(self: ClusterBuilder, z: f32) u32 {
        const zc = @max(z, self.near);
        const s = @log(zc) * self.slice_scale + self.slice_bias;
        const si: i32 = @intFromFloat(@floor(s));
        return @intCast(std.math.clamp(si, 0, @as(i32, @intCast(self.grid.z)) - 1));
    }

    /// Inverse of sliceForDepth: the view depth at a slice's near plane.
    ///
    /// Currently unused. The old screen-bound code called it to re-project each
    /// light per slice; the tangent bound is the sphere's silhouette from the
    /// eye and does not vary with depth, so it is now computed once per light.
    /// Kept because it is the exact inverse of sliceForDepth and any future
    /// per-slice refinement (clipping the sphere to a slice for a tighter bound)
    /// needs it.
    fn depthForSlice(self: ClusterBuilder, slice: u32) f32 {
        const s: f32 = @floatFromInt(slice);
        return @exp((s - self.slice_bias) / self.slice_scale);
    }

    fn upload(self: *ClusterBuilder) void {
        var table_data = sg.ImageData{};
        table_data.mip_levels[0] = sg.asRange(self.table.items);
        sg.updateImage(self.table_tex, table_data);

        // The index texture is sized in whole texels; pad the tail so the upload
        // covers the full image (sokol requires the complete surface).
        const needed: usize = @intCast(self.index_tex_texels * 4);
        if (self.indices.items.len < needed) {
            self.indices.resize(self.allocator, needed) catch return;
            @memset(self.indices.items[self.grid.total() * self.capacity ..], 0);
        }
        var index_data = sg.ImageData{};
        index_data.mip_levels[0] = sg.asRange(self.indices.items[0..needed]);
        sg.updateImage(self.index_tex, index_data);
    }

    pub fn tableView(self: ClusterBuilder) sg.View {
        return self.table_view;
    }

    pub fn indexListView(self: ClusterBuilder) sg.View {
        return self.index_view;
    }
};

// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------

fn ndcToTile(ndc: f32, tiles: u32) i32 {
    const t: f32 = @floatFromInt(tiles);
    const v = (ndc * 0.5 + 0.5) * t;
    return @intFromFloat(@floor(v));
}

/// Transform a world point into view space. Row-vector convention (p * M),
/// matching zmath's layout as used throughout the renderer.
fn transformPoint(m: math.Matrix, p: Vec3) Vec3 {
    const zm = @import("zmath");
    const v = zm.f32x4(p.x, p.y, p.z, 1.0);
    const r = zm.mul(v, m);
    return .{ .x = r[0], .y = r[1], .z = r[2] };
}
