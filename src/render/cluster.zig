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
    table: std.ArrayListUnmanaged([2]u32) = .empty, // (offset, count) per froxel
    indices: std.ArrayListUnmanaged(u32) = .empty, // flat, capacity per froxel
    counts: std.ArrayListUnmanaged(u32) = .empty, // fill count per froxel
    bounds: std.ArrayListUnmanaged(LightBound) = .empty,

    /// Depth-slice constants, recomputed when the camera's near/far change.
    /// Uploaded to the shader so it derives the same slice index.
    slice_scale: f32 = 0,
    slice_bias: f32 = 0,
    near: f32 = 0,
    far: f32 = 0,

    index_tex_texels: u32 = 0,

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
        @memset(self.counts.items, 0);
        @memset(self.indices.items, 0);

        // Fixed slots: froxel i owns indices [i*capacity, (i+1)*capacity).
        for (self.table.items, 0..) |*e, i| {
            e.* = .{ @as(u32, @intCast(i)) * self.capacity, 0 };
        }

        if (lights.len > 0) {
            self.computeBounds(lights, camera);
            self.assign(camera);
        }

        // Publish the fill counts.
        for (self.table.items, 0..) |*e, i| {
            e.*[1] = self.counts.items[i];
        }

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

    fn assign(self: *ClusterBuilder, camera: Camera3D) void {
        const gx = self.grid.x;
        const gy = self.grid.y;
        const gz = self.grid.z;

        // Tile extents in view space at z=1, from the projection.
        const tan_half_fov = @tan(camera.fov_y * 0.5);
        const aspect = camera.aspect;

        for (self.bounds.items, 0..) |b, li| {
            // View-space depth range the sphere spans (LH: +z forward).
            const z_min = @max(b.center_vs.z - b.radius, camera.near);
            const z_max = @min(b.center_vs.z + b.radius, camera.far);
            if (z_max <= camera.near or z_min >= camera.far) continue;

            const slice_min = self.sliceForDepth(z_min);
            const slice_max = self.sliceForDepth(z_max);

            var sz = slice_min;
            while (sz <= slice_max and sz < gz) : (sz += 1) {
                // Use the near plane of this slice for the tightest screen bound
                // the sphere could occupy within it.
                const slice_z = @max(self.depthForSlice(sz), camera.near);
                const eff_z = @max(slice_z, b.center_vs.z - b.radius);
                if (eff_z <= 0) continue;

                // Project the sphere to a screen-space AABB at this depth.
                const half_h = tan_half_fov * eff_z;
                const half_w = half_h * aspect;
                if (half_w <= 0 or half_h <= 0) continue;

                const x_min_ndc = (b.center_vs.x - b.radius) / half_w;
                const x_max_ndc = (b.center_vs.x + b.radius) / half_w;
                const y_min_ndc = (b.center_vs.y - b.radius) / half_h;
                const y_max_ndc = (b.center_vs.y + b.radius) / half_h;

                // NDC [-1,1] -> tile range.
                const tx_min = ndcToTile(x_min_ndc, gx);
                const tx_max = ndcToTile(x_max_ndc, gx);
                const ty_min = ndcToTile(y_min_ndc, gy);
                const ty_max = ndcToTile(y_max_ndc, gy);
                if (tx_max < 0 or ty_max < 0) continue;

                var ty: i32 = @max(ty_min, 0);
                while (ty <= ty_max and ty < @as(i32, @intCast(gy))) : (ty += 1) {
                    var tx: i32 = @max(tx_min, 0);
                    while (tx <= tx_max and tx < @as(i32, @intCast(gx))) : (tx += 1) {
                        const froxel = (sz * gy + @as(u32, @intCast(ty))) * gx + @as(u32, @intCast(tx));
                        const c = self.counts.items[froxel];
                        if (c >= self.capacity) continue; // full: drop
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
