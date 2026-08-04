//! src/render/light.zig
//!
//! Lights + the CPU->GPU light store shared by the forward (mesh.glsl) and
//! deferred (deferred_lighting.glsl) paths.
//!
//! CLUSTERED LIGHTING — the light set no longer lives in a uniform array with a
//! hard 16-light cap. Instead:
//!
//!   * DIRECTIONAL lights stay in a small uniform block. They affect every pixel,
//!     so clustering them would waste index-list space on data that's identical
//!     everywhere. A handful is all any scene needs.
//!
//!   * PUNCTUAL lights (point/spot, and later area) live in a FLOAT TEXTURE,
//!     read with texelFetch. This is deliberately a texture and not a storage
//!     buffer: WebGL2 and GLES 3.0 have float textures and texelFetch in core,
//!     but no SSBOs. Keeping the shader-side read path identical everywhere
//!     means pbr_lib.glsl.inc has exactly ONE variant — which matters enormously
//!     once shadows, and everything after, are written against it.
//!
//! HANDLES: users hold a stable LightHandle, not an index. Internal storage is
//! kept packed and sorted by type (so cluster index lists come out naturally
//! type-grouped at zero cost, and the shader's type branch stays coherent across
//! a wavefront). Sorting happens on mutation rather than per frame; a
//! handle->index map absorbs the reordering.
//!
//! TEXTURE LAYOUT (the CPU<->GPU contract — changing it means changing
//! pbr_lib.glsl.inc's accessors in lockstep):
//!
//!   RGBA32F, texels_per_light = 6, lights laid out along rows:
//!     texel 0: position.xyz,   type
//!     texel 1: direction.xyz,  range
//!     texel 2: color.rgb,      intensity
//!     texel 3: cos(inner), cos(outer), 0, 0
//!     texel 4: tangent.xyz,    half_width      <- area lights (reserved)
//!     texel 5: bitangent.xyz,  half_height     <- area lights (reserved)
//!
//! Texels 4-5 are zero for point/spot today. They're allocated NOW because area
//! lights (a rect light matching an emissive panel is how engines make glowing
//! geometry actually illuminate) need them, and widening the layout later would
//! mean re-churning the packer, the cluster builder, and the shader together.

const std = @import("std");
const sg = @import("sokol").gfx;
const math = @import("../math.zig");
const Vec3 = math.Vec3;
const DEG_TO_RAD = math.DEG_TO_RAD;
const Color = @import("../root.zig").Color;

/// Area types are declared but not yet evaluated by the shader — the layout
/// reserves their storage so adding them is shader work only.
pub const LightType = enum(u32) {
    directional = 0,
    point = 1,
    spot = 2,
    // rect = 3,  // future: LTC-evaluated rect light
    // tube = 4,  // future: LTC-evaluated tube/line light
};

pub const Light = struct {
    type: LightType = .directional,
    /// Travel direction (sun pointing down = {0,-1,0}; spot cone axis).
    direction: Vec3 = .{ .x = -0.4, .y = -1.0, .z = -0.3 },
    /// World position (point/spot only).
    position: Vec3 = .{ .x = 0, .y = 0, .z = 0 },
    color: Color = .{ .r = 1, .g = 1, .b = 1, .a = 1 },
    intensity: f32 = 3.0,
    /// Falloff radius for point/spot. Doubles as the CULLING BOUND for
    /// clustering — a light is only assigned to froxels within this distance, so
    /// an over-large range costs real performance, not just visual reach.
    range: f32 = 20.0,
    /// Spot cone half-angles in degrees: full intensity within inner, fading to
    /// zero at outer.
    spot_inner_deg: f32 = 20.0,
    spot_outer_deg: f32 = 30.0,

    /// Area-light extent (rect/tube). Unused by point/spot; reserved so the GPU
    /// layout doesn't change when area lights land.
    tangent: Vec3 = .{ .x = 1, .y = 0, .z = 0 },
    bitangent: Vec3 = .{ .x = 0, .y = 1, .z = 0 },
    half_width: f32 = 0,
    half_height: f32 = 0,

    pub fn directional(dir: Vec3, color: Color, intensity: f32) Light {
        return .{ .type = .directional, .direction = dir, .color = color, .intensity = intensity };
    }

    /// Point lights use inverse-square falloff, so they generally need a higher
    /// intensity than directional lights to read at a distance.
    pub fn point(position: Vec3, color: Color, intensity: f32, range: f32) Light {
        return .{ .type = .point, .position = position, .color = color, .intensity = intensity, .range = range };
    }

    pub fn spot(position: Vec3, direction: Vec3, color: Color, intensity: f32, range: f32, inner_deg: f32, outer_deg: f32) Light {
        return .{
            .type = .spot,
            .position = position,
            .direction = direction,
            .color = color,
            .intensity = intensity,
            .range = range,
            .spot_inner_deg = inner_deg,
            .spot_outer_deg = outer_deg,
        };
    }

    /// True for lights that get clustered. Directional lights are evaluated by
    /// every pixel from the uniform block instead.
    pub fn isPunctual(self: Light) bool {
        return self.type != .directional;
    }
};

/// Stable, user-facing reference to a light. Survives the internal re-sorting
/// that keeps storage packed by type.
pub const LightHandle = struct {
    id: u32,

    pub const invalid_id: u32 = std.math.maxInt(u32);
    pub const invalid = LightHandle{ .id = invalid_id };

    pub fn isValid(self: LightHandle) bool {
        return self.id != invalid_id;
    }
};

// ---------------------------------------------------------------------------
//  GPU layout constants — mirrored in pbr_lib.glsl.inc.
// ---------------------------------------------------------------------------

/// RGBA32F texels per punctual light. 6 (not 4) so area-light extents have a
/// home from day one.
pub const texels_per_light: usize = 6;

/// Punctual lights per texture row. Keeps the texture width at
/// lights_per_row * texels_per_light = 1536, well under the 16384 max dimension,
/// while a 4096-light store is only ~16 rows tall.
pub const lights_per_row: usize = 256;

/// Directional lights evaluated by every pixel. Kept small on purpose: each one
/// costs a full shading evaluation for every shaded pixel on screen.
pub const max_directional: usize = 4;

/// std140 uniform block for the always-on lighting state. Field order and types
/// MUST match the generated LightParams in mesh.glsl.zig / lambert.glsl.zig /
/// deferred_lighting.glsl.zig - all three are byte-identical, which is why one
/// packing function can feed every path.
///
/// Note what's NOT here any more: the 16-entry punctual arrays, and the per-draw
/// re-upload they forced in the forward path (they used to live inside mesh's
/// fs_params, so every mesh re-sent the whole light set).
pub const GlobalLightParams = extern struct {
    camera_pos: [4]f32 align(16),
    /// rgb = ambient, w = directional light count
    ambient_count: [4]f32 align(1),
    /// x = punctual light count, y = light texture width (texels),
    /// z = cluster capacity, w = unused
    light_info: [4]f32 align(1),
    /// x = slice scale, y = slice bias, z = grid.z, w = unused
    cluster_slice: [4]f32 align(1),
    /// x = grid.x, y = grid.y, z = screen width, w = screen height
    cluster_grid: [4]f32 align(1),
    /// x = near, y = far, z = origin_top_left flag, w = unused
    cluster_depth: [4]f32 align(1),
    dir_direction: [max_directional][4]f32 align(1), // xyz travel direction
    dir_color: [max_directional][4]f32 align(1), // rgb, w = intensity
};

// ---------------------------------------------------------------------------
//  LightStore — owns the light set, the handle map, and the GPU texture.
// ---------------------------------------------------------------------------

pub const LightStore = struct {
    allocator: std.mem.Allocator,

    /// Packed and sorted by type. Indices here are NOT stable across mutation.
    lights: std.ArrayListUnmanaged(Light) = .empty,
    /// Parallel to `lights`: which handle owns each slot. Used to patch the map
    /// after a re-sort.
    owners: std.ArrayListUnmanaged(u32) = .empty,
    /// handle id -> index into `lights`. Sparse; invalid entries are invalid_id.
    handle_to_index: std.ArrayListUnmanaged(u32) = .empty,
    /// Recycled handle ids, so long-running scenes don't grow the map forever.
    free_handles: std.ArrayListUnmanaged(u32) = .empty,

    /// Index of the first punctual light in `lights`. Everything before it is
    /// directional. Maintained by the sort.
    punctual_start: usize = 0,

    /// Set when the light set changes; drives re-sort + re-upload.
    dirty: bool = true,

    // GPU side
    tex: sg.Image = .{},
    view: sg.View = .{},
    tex_capacity: usize = 0, // lights the current texture can hold
    staging: std.ArrayListUnmanaged([4]f32) = .empty,

    pub fn init(allocator: std.mem.Allocator) LightStore {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *LightStore) void {
        self.lights.deinit(self.allocator);
        self.owners.deinit(self.allocator);
        self.handle_to_index.deinit(self.allocator);
        self.free_handles.deinit(self.allocator);
        self.staging.deinit(self.allocator);
        self.destroyTexture();
    }

    fn destroyTexture(self: *LightStore) void {
        if (self.view.id != 0) sg.destroyView(self.view);
        if (self.tex.id != 0) sg.destroyImage(self.tex);
        self.view = .{};
        self.tex = .{};
        self.tex_capacity = 0;
    }

    // -----------------------------------------------------------------------
    //  Mutation
    // -----------------------------------------------------------------------

    pub fn add(self: *LightStore, light: Light) !LightHandle {
        const id = if (self.free_handles.pop()) |recycled| recycled else blk: {
            const new_id: u32 = @intCast(self.handle_to_index.items.len);
            try self.handle_to_index.append(self.allocator, LightHandle.invalid_id);
            break :blk new_id;
        };

        try self.lights.append(self.allocator, light);
        try self.owners.append(self.allocator, id);
        self.handle_to_index.items[id] = @intCast(self.lights.items.len - 1);
        self.dirty = true;
        return .{ .id = id };
    }

    pub fn remove(self: *LightStore, handle: LightHandle) void {
        const idx = self.indexOf(handle) orelse return;
        const last = self.lights.items.len - 1;

        // Swap-and-pop, then repair the moved light's map entry. The sort in
        // flush() restores type grouping, so order doesn't need preserving here.
        if (idx != last) {
            self.lights.items[idx] = self.lights.items[last];
            self.owners.items[idx] = self.owners.items[last];
            self.handle_to_index.items[self.owners.items[idx]] = @intCast(idx);
        }
        _ = self.lights.pop();
        _ = self.owners.pop();

        self.handle_to_index.items[handle.id] = LightHandle.invalid_id;
        self.free_handles.append(self.allocator, handle.id) catch {};
        self.dirty = true;
    }

    fn indexOf(self: *const LightStore, handle: LightHandle) ?usize {
        if (handle.id >= self.handle_to_index.items.len) return null;
        const idx = self.handle_to_index.items[handle.id];
        if (idx == LightHandle.invalid_id) return null;
        return idx;
    }

    /// Mutable access. Marks the store dirty: changing `type` needs the sort to
    /// re-run, and changing anything else needs the texture re-uploaded.
    pub fn get(self: *LightStore, handle: LightHandle) ?*Light {
        const idx = self.indexOf(handle) orelse return null;
        self.dirty = true;
        return &self.lights.items[idx];
    }

    pub fn getConst(self: *const LightStore, handle: LightHandle) ?*const Light {
        const idx = self.indexOf(handle) orelse return null;
        return &self.lights.items[idx];
    }

    pub fn clear(self: *LightStore) void {
        self.lights.clearRetainingCapacity();
        self.owners.clearRetainingCapacity();
        for (self.handle_to_index.items) |*e| e.* = LightHandle.invalid_id;
        self.free_handles.clearRetainingCapacity();
        self.punctual_start = 0;
        self.dirty = true;
    }

    pub fn count(self: *const LightStore) usize {
        return self.lights.items.len;
    }

    /// Directional lights — evaluated by every pixel, packed into the uniform.
    pub fn directionalLights(self: *const LightStore) []const Light {
        return self.lights.items[0..self.punctual_start];
    }

    /// Punctual lights (point/spot/area) — clustered, read from the texture.
    /// Cluster index i refers to punctual light i, NOT to the global list.
    pub fn punctualLights(self: *const LightStore) []const Light {
        return self.lights.items[self.punctual_start..];
    }

    // -----------------------------------------------------------------------
    //  Upload
    // -----------------------------------------------------------------------

    /// Re-sort and re-upload if anything changed. Call once per frame before
    /// cluster building; it's a no-op when the light set is static.
    pub fn flush(self: *LightStore) void {
        if (!self.dirty) return;
        self.sortByType();
        self.uploadTexture();
        self.dirty = false;
    }

    /// Sort by type so directional lights lead and punctual lights are grouped.
    /// Two payoffs: directionalLights()/punctualLights() become slices instead of
    /// filters, and cluster index lists come out type-grouped for free (the
    /// builder appends in ascending index order), keeping the shader's type
    /// branch coherent without a separate per-cluster sort.
    fn sortByType(self: *LightStore) void {
        const Ctx = struct {
            lights: []Light,
            owners: []u32,

            pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                return @intFromEnum(ctx.lights[a].type) < @intFromEnum(ctx.lights[b].type);
            }
            pub fn swap(ctx: @This(), a: usize, b: usize) void {
                std.mem.swap(Light, &ctx.lights[a], &ctx.lights[b]);
                std.mem.swap(u32, &ctx.owners[a], &ctx.owners[b]);
            }
        };
        std.sort.insertionContext(0, self.lights.items.len, Ctx{
            .lights = self.lights.items,
            .owners = self.owners.items,
        });

        // Repair the handle map after reordering.
        for (self.owners.items, 0..) |owner, i| {
            self.handle_to_index.items[owner] = @intCast(i);
        }

        // First non-directional light.
        self.punctual_start = self.lights.items.len;
        for (self.lights.items, 0..) |l, i| {
            if (l.isPunctual()) {
                self.punctual_start = i;
                break;
            }
        }
    }

    fn uploadTexture(self: *LightStore) void {
        const punctual = self.punctualLights();

        // Always keep a VALID texture even with zero punctual lights: the shader
        // unconditionally declares and binds light_data, and binding an invalid
        // view is a validation error. One row is the floor.
        const rows_needed = @max(1, (punctual.len + lights_per_row - 1) / lights_per_row);
        const capacity = rows_needed * lights_per_row;
        if (capacity > self.tex_capacity) {
            self.destroyTexture();
            const w: i32 = @intCast(lights_per_row * texels_per_light);
            const h: i32 = @intCast(rows_needed);
            self.tex = sg.makeImage(.{
                .width = w,
                .height = h,
                .pixel_format = .RGBA32F,
                .usage = .{ .stream_update = true },
                .label = "light_data",
            });
            self.view = sg.makeView(.{ .texture = .{ .image = self.tex } });
            self.tex_capacity = capacity;
        }

        // Pack. Row-major, texels_per_light texels per light.
        const total_texels = self.tex_capacity * texels_per_light;
        self.staging.resize(self.allocator, total_texels) catch return;
        @memset(self.staging.items, [4]f32{ 0, 0, 0, 0 });

        for (punctual, 0..) |l, i| {
            const row = i / lights_per_row;
            const col = i % lights_per_row;
            const base = row * (lights_per_row * texels_per_light) + col * texels_per_light;

            const t: f32 = @floatFromInt(@intFromEnum(l.type));
            self.staging.items[base + 0] = .{ l.position.x, l.position.y, l.position.z, t };
            self.staging.items[base + 1] = .{ l.direction.x, l.direction.y, l.direction.z, l.range };
            self.staging.items[base + 2] = .{ l.color.r, l.color.g, l.color.b, l.intensity };
            self.staging.items[base + 3] = .{
                @cos(l.spot_inner_deg * DEG_TO_RAD),
                @cos(l.spot_outer_deg * DEG_TO_RAD),
                0,
                0,
            };
            // Reserved for area lights; zero for point/spot.
            self.staging.items[base + 4] = .{ l.tangent.x, l.tangent.y, l.tangent.z, l.half_width };
            self.staging.items[base + 5] = .{ l.bitangent.x, l.bitangent.y, l.bitangent.z, l.half_height };
        }

        var data = sg.ImageData{};
        data.mip_levels[0] = sg.asRange(self.staging.items);
        sg.updateImage(self.tex, data);
    }

    /// Uniform block for the always-on lighting state. The cluster fields come
    /// from the ClusterBuilder, so this is called after cluster.build().
    ///
    /// `origin_top_left` must be sg.queryFeatures().origin_top_left: gl_FragCoord's
    /// y origin differs across backends, but the CPU assigns froxels in NDC
    /// (+y up), so the shader has to know whether to flip. Getting this wrong
    /// mirrors the light assignment vertically - which looks almost right and is
    /// miserable to diagnose.
    pub fn packGlobals(
        self: *const LightStore,
        ambient: Color,
        camera_pos: Vec3,
        camera_near: f32,
        camera_far: f32,
        screen_w: f32,
        screen_h: f32,
        cluster_slice: [4]f32,
        cluster_grid_xy: [2]f32,
        cluster_capacity: u32,
        origin_top_left: bool,
    ) GlobalLightParams {
        var p: GlobalLightParams = std.mem.zeroes(GlobalLightParams);
        p.camera_pos = .{ camera_pos.x, camera_pos.y, camera_pos.z, 0 };

        const dirs = self.directionalLights();
        const n = @min(dirs.len, max_directional);
        p.ambient_count = .{ ambient.r, ambient.g, ambient.b, @floatFromInt(n) };
        p.light_info = .{
            @floatFromInt(self.punctualLights().len),
            @floatFromInt(lights_per_row * texels_per_light),
            @floatFromInt(cluster_capacity),
            0,
        };
        p.cluster_slice = cluster_slice;
        p.cluster_grid = .{ cluster_grid_xy[0], cluster_grid_xy[1], screen_w, screen_h };
        p.cluster_depth = .{ camera_near, camera_far, if (origin_top_left) 1.0 else 0.0, 0 };

        for (dirs[0..n], 0..) |l, i| {
            p.dir_direction[i] = .{ l.direction.x, l.direction.y, l.direction.z, 0 };
            p.dir_color[i] = .{ l.color.r, l.color.g, l.color.b, l.intensity };
        }
        return p;
    }
};
