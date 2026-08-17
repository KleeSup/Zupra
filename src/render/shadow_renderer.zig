//! src/render/shadow_renderer.zig
//!
//! Builds shadow CASTERS from lights (light-space matrices + atlas tiles) and
//! renders the depth-only passes into the atlas.
//!
//! Phase 1: directional light, single map or N cascades, fit to the camera
//! frustum within max_distance. Spot/point are added later as more caster
//! builders — the renderer loop over casters never changes.

const std = @import("std");
const sg = @import("sokol").gfx;
const math = @import("../math.zig");
const zm = math.zm;
const shadow = @import("shadow.zig");
const light_mod = @import("light.zig");
const pipeline = @import("../graphics/pipeline.zig");
const mesh_mod = @import("mesh.zig");
const model_mod = @import("model.zig");
const Camera3D = @import("camera3d.zig").Camera3D;
const zupra = @import("../root.zig");

const shd_depth = @import("shaders").shadow_depth;
const shd_depth_inst = @import("shaders").shadow_depth_instanced;

const Matrix = math.Matrix;
const Vec3 = math.Vec3;
const Light = light_mod.Light;
const ShadowAtlas = shadow.ShadowAtlas;
const ShadowCaster = shadow.ShadowCaster;
const Frustum = @import("culling.zig").Frustum;
const ShadowSettings = shadow.ShadowSettings;
const AtlasRect = shadow.AtlasRect;
const Mesh = mesh_mod.Mesh;
const Material = @import("material.zig").Material;
const ModelInstance = model_mod.ModelInstance;
const PipelineCache = pipeline.PipelineCache;
const PipelineKey = pipeline.PipelineKey;
const PassSignature = pipeline.PassSignature;

/// Max cascades per directional light. 4 is the industry norm.
pub const max_cascades = 4;

/// Shadow VIEWS per light -- the stride of the per-light matrix and rect
/// arrays. Six, because a point light needs one map per cube face; a
/// directional uses up to max_cascades of them and a spot exactly one.
///
/// Sized by the largest consumer rather than per type because the uniform block
/// is a flat array with a fixed stride: sh_vp[light * MAX_SHADOW_VIEWS + view].
/// A ragged layout would need an offset table and another indirection in the
/// shader to save a few kilobytes of uniform space.
pub const max_shadow_views = 6;

/// Instance-buffer capacity, in bytes: 16384 model matrices at 64 bytes each.
/// The whole frame's batched shadow draws sub-allocate from this one buffer,
/// so it has to cover queue length times view count, not just queue length.
const max_instance_bytes = 16384 * 16 * @sizeOf(f32);
/// Max simultaneously shadowed lights (any type). The shadow_data shader array
/// is sized to this. Generous, but each entry is small.
pub const max_shadowed = 16;

/// One shadowed light's data, uploaded parallel to the light list. The shader
/// indexes this by the light's shadow_index (stored in the light texture).
pub const ShadowData = extern struct {
    /// Light-space view-projection per VIEW: cascade 0..n for a directional,
    /// [0] alone for a spot, cube face 0..5 for a point light.
    view_proj: [max_shadow_views][16]f32,
    /// Atlas rect per view (xy origin, zw size, in [0,1] UV).
    rect: [max_shadow_views][4]f32,
    /// x = VIEW count (cascades for directional, 1 for spot, 6 for point),
    /// y = pcf radius (texels), z = cascade blend fraction,
    /// w = atlas texel size (1/atlas_size, for PCF tap spacing).
    params: [4]f32,
    /// Cascade split distances (view-space), x..w = splits 0..3 far planes.
    /// Used to pick the cascade for a fragment by its depth.
    splits: [4]f32,
    /// Normal-offset bias in WORLD units, one per cascade (x..w = cascades
    /// 0..3). Resolved on the CPU because it depends on each view's fitted
    /// extent, which the shader has no way to recover from the matrix.
    ///
    /// Four components for up to six views: spot and point lights write the
    /// same value to all four, since every face of a cube shares one fov and one
    /// range, so whichever component the shader reads is the right one.
    normal_bias: [4]f32,
    /// Distance fade at the far end of the shadow range, as
    /// x = view depth where the fade starts, y = 1 / (end - start) so the shader
    /// multiplies instead of dividing, z and w reserved.
    ///
    /// y = 0 disables the fade, which is how spot and point lights opt out. The
    /// two reserved components follow the same reasoning as the unused texels in
    /// light.zig: per-light shadow controls (strength, normal-bias override) are
    /// the obvious next additions here, and widening a GPU-side layout later
    /// means re-churning the packer and all three shader blocks together.
    fade: [4]f32,
    /// xyz = light world position, w = 1.0 for a cube (point) light, 0.0
    /// otherwise. The shader needs the position to work out which cube face a
    /// fragment falls on, and the flag to know to look at all -- a directional
    /// has neither a position nor faces.
    pos_kind: [4]f32,
};

/// A point or spot light's cached shadow tiles.
///
/// Reservation and validity are deliberately separate. The rectangle belongs to
/// the light for as long as the entry lives, whether or not the depth currently
/// in it is correct, so regenerating a shadow never costs it its place in the
/// atlas. Tiles that moved on every regeneration would defeat the point of
/// caching.
const CacheEntry = struct {
    /// Identifies the light this belongs to across frames.
    light_id: u64,
    /// One per view: a spot uses [0], a point uses all six. Held from allocation
    /// until the entry is evicted.
    tiles: [max_shadow_views]ShadowAtlas.Tile = undefined,
    view_count: u32 = 0,

    /// Whether the depth in those tiles is still correct. False forces a redraw
    /// into the same rectangles.
    valid: bool = false,

    /// The inputs the cached depth was generated from. Any change invalidates
    /// it, since the recorded depth would no longer describe the scene.
    light_position: Vec3 = .{ .x = 0, .y = 0, .z = 0 },
    light_direction: Vec3 = .{ .x = 0, .y = 0, .z = 0 },
    light_range: f32 = 0,
    spot_outer_deg: f32 = 0,
    resolution: u32 = 0,

    /// Which frame this entry was last asked for. Entries not touched for a
    /// frame belong to lights that are gone or no longer shadowed, and their
    /// tiles are released.
    last_used_frame: u64 = 0,
};

pub const ShadowRenderer = struct {
    cache: *PipelineCache,
    atlas: ShadowAtlas,
    shader: sg.Shader,

    /// Cached static shadows for point and spot lights, keyed by light id.
    ///
    /// Directional lights are excluded for now. Their views are fitted to the
    /// camera frustum, so they invalidate on camera movement, and handling that
    /// well needs the fit itself to report whether it changed.
    shadow_cache: std.ArrayListUnmanaged(CacheEntry) = .empty,
    /// Off by default. Caching trades atlas space and complexity for skipped
    /// draws, and a scene with few static lights gains little.
    caching_enabled: bool = false,
    frame_index: u64 = 0,

    /// Views whose cached depth is still valid this frame, so the depth pass
    /// skips them entirely. Indices into `casters`.
    cached_view: std.ArrayListUnmanaged(bool) = .empty,

    /// Casters built this frame (all lights, all cascades/faces).
    casters: std.ArrayListUnmanaged(ShadowCaster) = .empty,

    /// Instanced depth shader + the streaming buffer its per-instance matrices
    /// come from. One buffer for the whole frame, sub-allocated per draw with
    /// sg.appendBuffer -- sokol allows a buffer only ONE updateBuffer per frame,
    /// but any number of appends, which is exactly the shape of "many small
    /// uploads across many passes".
    inst_shader: sg.Shader = .{},
    inst_buf: sg.Buffer = .{},
    /// Per-shadowed-light data, uploaded to the lighting shader.
    data: [max_shadowed]ShadowData = undefined,
    data_count: u32 = 0,

    allocator: std.mem.Allocator,

    /// Returns an error now, because the atlas allocates its tile tree.
    pub fn init(allocator: std.mem.Allocator, cache: *PipelineCache, opts: shadow.AtlasOptions) !ShadowRenderer {
        const inst_shader = sg.makeShader(shd_depth_inst.shadowDepthInstancedShaderDesc(sg.queryBackend()));
        // Sized for the worst realistic frame: every queued caster drawn into
        // every view. Overflowing is not a crash -- sokol flags the buffer and
        // drops the appends -- but it silently loses shadows, so the check in
        // drawMeshInstanced reports it instead of failing quietly.
        const inst_buf = sg.makeBuffer(.{
            .size = max_instance_bytes,
            .usage = .{ .vertex_buffer = true, .stream_update = true },
        });
        const shader = sg.makeShader(shd_depth.shadowDepthShaderDesc(sg.queryBackend()));
        // Diagnostic: if the depth-only shader failed to compile, every shadow
        // draw will die at apply_uniforms with "pipeline no longer alive". Catch
        // it here where the cause is obvious.
        const state = sg.queryShaderState(shader);
        if (state != .VALID) {
            std.log.err("ShadowRenderer: shadow_depth shader is {s} (not VALID) — depth pass will fail", .{@tagName(state)});
        }
        const inst_state = sg.queryShaderState(inst_shader);
        if (inst_state != .VALID) {
            std.log.err("ShadowRenderer: shadow_depth_instanced shader is {s} (not VALID) — batched depth draws will fail", .{@tagName(inst_state)});
        }
        return .{
            .cache = cache,
            .atlas = try ShadowAtlas.init(allocator, opts),
            .shader = shader,
            .inst_shader = inst_shader,
            .inst_buf = inst_buf,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ShadowRenderer) void {
        self.casters.deinit(self.allocator);
        self.shadow_cache.deinit(self.allocator);
        self.cached_view.deinit(self.allocator);
        self.atlas.deinit();
        sg.destroyBuffer(self.inst_buf);
        sg.destroyShader(self.inst_shader);
        sg.destroyShader(self.shader);
    }

    /// Find or create this light's cache entry, and decide whether its contents
    /// are still usable.
    ///
    /// Returns null when caching is off or the atlas cannot hold the tiles, and
    /// the caller then falls back to ordinary transient allocation.
    fn acquireCache(
        self: *ShadowRenderer,
        light_id: u64,
        l: Light,
        s: ShadowSettings,
        view_count: u32,
    ) ?*CacheEntry {
        if (!self.caching_enabled) return null;

        var entry: ?*CacheEntry = null;
        for (self.shadow_cache.items) |*e| {
            if (e.light_id == light_id) {
                entry = e;
                break;
            }
        }

        if (entry == null) {
            // Persistent tiles for every view up front. Partial allocation would
            // leave a point light cached in some directions and not others,
            // which reads as geometry randomly failing to cast.
            var fresh = CacheEntry{ .light_id = light_id, .view_count = view_count };
            var i: u32 = 0;
            while (i < view_count) : (i += 1) {
                const tile = self.atlas.allocatePersistent(s.resolution) orelse {
                    // Roll back what was taken, so a failed light does not hold
                    // atlas space it cannot use.
                    var j: u32 = 0;
                    while (j < i) : (j += 1) self.atlas.freeTile(fresh.tiles[j].handle);
                    return null;
                };
                fresh.tiles[i] = tile;
            }
            self.shadow_cache.append(self.allocator, fresh) catch return null;
            entry = &self.shadow_cache.items[self.shadow_cache.items.len - 1];
        }

        const e = entry.?;
        e.last_used_frame = self.frame_index;

        // Invalidate on any change to the inputs the depth was generated from.
        // Resolution is included because a different tile size means different
        // rectangles entirely, not merely different contents.
        const moved = !vecEql(e.light_position, l.position) or
            !vecEql(e.light_direction, l.direction) or
            e.light_range != l.range or
            e.spot_outer_deg != l.spot_outer_deg or
            e.resolution != s.resolution or
            e.view_count != view_count;

        if (moved) {
            e.valid = false;
            e.light_position = l.position;
            e.light_direction = l.direction;
            e.light_range = l.range;
            e.spot_outer_deg = l.spot_outer_deg;
            e.resolution = s.resolution;
            // A change of view count or resolution means the tiles themselves
            // are wrong, not just their contents.
            if (e.view_count != view_count or e.resolution != s.resolution) {
                var i: u32 = 0;
                while (i < e.view_count) : (i += 1) self.atlas.freeTile(e.tiles[i].handle);
                e.view_count = 0;
                return null; // reallocated next frame, transient this one
            }
        }

        return e;
    }

    /// Release tiles for lights that were not asked about this frame, which
    /// means they were removed or stopped casting shadows.
    fn evictStaleCache(self: *ShadowRenderer) void {
        var i: usize = 0;
        while (i < self.shadow_cache.items.len) {
            const e = &self.shadow_cache.items[i];
            if (e.last_used_frame == self.frame_index) {
                i += 1;
                continue;
            }
            var v: u32 = 0;
            while (v < e.view_count) : (v += 1) self.atlas.freeTile(e.tiles[v].handle);
            _ = self.shadow_cache.swapRemove(i);
        }
    }

    /// Invalidate every cached shadow. For when the scene changed in a way the
    /// per-light checks cannot see, such as a static caster being added, moved
    /// or removed.
    ///
    /// A static light and a static camera do not imply a static shadow: an
    /// object entering or leaving a light's volume changes the depth while
    /// nothing about the light changes. Until caster-scene versioning exists,
    /// this is the caller's way to say so.
    pub fn invalidateCache(self: *ShadowRenderer) void {
        for (self.shadow_cache.items) |*e| e.valid = false;
    }

    /// PassSignature for the depth-only atlas passes: no color, depth only.
    fn depthPassSignature(self: ShadowRenderer) PassSignature {
        _ = self;
        return .{ .color_count = 0, .depth_format = .DEPTH, .sample_count = 1 };
    }

    // -----------------------------------------------------------------------
    //  Build casters from lights (CPU)
    // -----------------------------------------------------------------------

    /// Rebuild the frame's casters + shadow data. Assigns each shadowed light a
    /// shadow_index; returns a slice mapping light -> shadow_index for the light
    /// store to bake into the light texture (so the shader can look it up).
    ///
    /// Phase 1 handles DIRECTIONAL lights. Spot/point builders slot in here as
    /// additional branches, each appending casters + one ShadowData.
    /// `owners` is the light store's parallel array of stable handle ids, one
    /// per light, and may be empty when the caller has none to give.
    ///
    /// The cache is keyed on those ids rather than on a light's index, because
    /// LightStore.remove swap-and-pops and flush() re-sorts by type, so an index
    /// identifies a different light from one frame to the next. Keying on it
    /// would hand one light another's cached tile, which looks correct until it
    /// does not. Without owners, caching is simply skipped rather than done
    /// unsafely.
    pub fn build(self: *ShadowRenderer, lights: []Light, owners: []const u32, camera: Camera3D) void {
        self.frame_index +%= 1;
        self.casters.clearRetainingCapacity();
        self.cached_view.clearRetainingCapacity();
        self.data_count = 0;
        self.atlas.reset();

        for (lights, 0..) |*l, light_index| {
            l.shadow_index = -1; // default: unshadowed this frame
            if (!l.shadow.enabled) continue;
            if (self.data_count >= max_shadowed) continue; // out of slots -> unshadowed

            const light_id: ?u64 = if (light_index < owners.len)
                owners[light_index]
            else
                null;

            const assigned: bool = switch (l.type) {
                .directional => self.buildDirectional(l.*, l.shadow, camera),
                .spot => self.buildSpot(l.*, l.shadow, camera, light_id),
                .point => self.buildPoint(l.*, l.shadow, camera),
            };
            if (assigned) {
                l.shadow_index = @intCast(self.data_count - 1);
            }
        }

        // Lights not seen this frame have gone or stopped casting, so their
        // reservations are released rather than held forever.
        if (self.caching_enabled) self.evictStaleCache();

        // Any tile still holding valid cached depth means the whole-atlas clear
        // would destroy it, so the renderer must clear per tile instead.
        self.atlas.has_cached_tiles = false;
        for (self.cached_view.items) |cached| {
            if (cached) {
                self.atlas.has_cached_tiles = true;
                break;
            }
        }
    }

    fn buildDirectional(self: *ShadowRenderer, l: Light, s: ShadowSettings, camera: Camera3D) bool {
        const cascades = std.math.clamp(s.cascade_count, 1, max_cascades);
        var d: ShadowData = std.mem.zeroes(ShadowData);
        d.params = .{
            @floatFromInt(cascades),
            @floatFromInt(s.pcf_radius),
            s.cascade_blend,
            1.0 / @as(f32, @floatFromInt(self.atlas.size)),
        };

        // Cascade split distances (practical logarithmic split): near..max_distance
        // partitioned so nearer cascades (which cover fewer world units per texel)
        // get proportionally more of the range. Single cascade = the whole range.
        var splits: [max_cascades + 1]f32 = undefined;
        computeSplits(camera.near, s.max_distance, cascades, &splits);
        d.splits = .{ splits[1], splits[2], splits[3], splits[4] };

        // No position and no cube faces: a directional light's rays are
        // parallel, so there is nothing for the shader to select between.
        d.pos_kind = .{ 0, 0, 0, 0 };

        // Fade over the last distance_fade of the range, reaching fully lit
        // exactly at max_distance -- which is also where the outermost cascade
        // stops having data, so the fade finishes precisely where the hard edge
        // would otherwise appear.
        const fade_end = s.max_distance;
        const fade_start = fade_end * (1.0 - std.math.clamp(s.distance_fade, 0.0, 1.0));
        const fade_span = fade_end - fade_start;
        d.fade = .{ fade_start, if (fade_span > 1e-4) 1.0 / fade_span else 0.0, 0, 0 };

        const light_dir = l.direction.normalize();

        var c: u32 = 0;
        while (c < cascades) : (c += 1) {
            const tile = self.atlas.allocate(s.resolution) orelse break; // atlas full
            const fit = fitDirectionalCascade(camera, light_dir, splits[c], splits[c + 1], tile.size, s.caster_extrusion);

            d.view_proj[c] = matToArr(fit.view_proj);
            d.rect[c] = tile.rect.uv;
            // Texels -> world, per cascade. A near cascade's texel is a fraction
            // of a far cascade's, so this is where one authored number becomes
            // four correct ones.
            d.normal_bias[c] = s.normal_bias_texels * fit.texel_world;

            self.casters.append(self.allocator, .{
                .view_proj = fit.view_proj,
                .rect = tile.rect,
                .tile_x = tile.x,
                .tile_y = tile.y,
                .tile_size = tile.size,
                .depth_bias = s.depth_bias,
                .depth_bias_slope = s.slope_bias,
                .cull = s.caster_cull,
                .frustum = Frustum.fromViewProj(fit.view_proj),
            }) catch {};
            // Directional cascades are not cached yet: they refit to the camera
            // frustum, so invalidation needs the fit itself to report whether it
            // changed. The flag still has to be appended to keep this array
            // index-aligned with casters.
            self.cached_view.append(self.allocator, false) catch {};
        }

        // If the atlas was full before even cascade 0 got a tile, this light is
        // unshadowed this frame (rect stays zero-size).
        if (d.rect[0][2] <= 0.0) return false;
        self.data[self.data_count] = d;
        self.data_count += 1;
        return true;
    }

    /// Spot lights need one tile and one perspective matrix -- the cone already
    /// defines the frustum, so there is no fitting problem to solve and no
    /// cascades to split. The shader side needs nothing new: sampleShadow
    /// already divides by w, so a projective matrix works there unchanged, and
    /// with cascade_count of 1 both pickCascade and the blend branch fall
    /// through to cascade 0.
    fn buildSpot(self: *ShadowRenderer, l: Light, s: ShadowSettings, camera: Camera3D, light_id: ?u64) bool {
        // Cheap reject: a spot whose reach never comes near the view isn't worth
        // a tile. Tiles are the scarce resource -- one wasted here is one a
        // visible light doesn't get -- so this is about allocation, not just
        // draw cost.
        const to_light = l.position.sub(camera.position).length();
        if (to_light - l.range > s.max_distance) return false;

        // A cached entry hands back the same rectangle every frame, so its depth
        // stays reusable. Falling back to a transient tile is always safe, just
        // not cached.
        const entry = if (light_id) |id| self.acquireCache(id, l, s, 1) else null;
        const tile = if (entry) |e| e.tiles[0] else (self.atlas.allocate(s.resolution) orelse return false);
        const reuse = if (entry) |e| e.valid else false;

        // The cone half-angle IS the frustum half-angle, so the vertical fov is
        // twice the outer angle at aspect 1. Clamped below a full hemisphere:
        // a perspective projection degenerates as the half-angle approaches 90
        // degrees, and a cone that wide wants a point light's cubemap anyway.
        const outer_deg = std.math.clamp(l.spot_outer_deg, 1.0, 89.0);
        const fov_y = std.math.degreesToRadians(outer_deg * 2.0);

        // Near plane scaled to the light's reach rather than fixed. Depth
        // precision in a projective map is governed by the far/near ratio, so a
        // hardcoded small near would quietly wreck a long-range spot while
        // costing nothing on a short one.
        const near_z = @max(0.05, l.range * 0.01);
        const far_z = @max(near_z + 0.1, l.range);

        const dir = l.direction.normalize();
        const up = pickUp(dir);
        const light_view = zm.lookAtLh(
            v4(l.position, 1),
            v4(l.position.add(dir), 1),
            v4(up, 0),
        );
        const light_proj = zm.perspectiveFovLh(fov_y, 1.0, near_z, far_z);
        const view_proj = zm.mul(light_view, light_proj);

        var d: ShadowData = std.mem.zeroes(ShadowData);
        d.params = .{
            1.0, // one cascade
            @floatFromInt(s.pcf_radius),
            0.0, // no cascade blending with a single cascade
            1.0 / @as(f32, @floatFromInt(self.atlas.size)),
        };
        d.splits = .{ far_z, far_z, far_z, far_z };
        d.view_proj[0] = matToArr(view_proj);
        d.rect[0] = tile.rect.uv;

        // A projective map's texel covers more world space the further it is
        // from the light, so unlike a cascade there is no single correct value.
        // Half the range is the compromise: referencing the far plane instead
        // would over-bias near the light and lift contact shadows off their
        // casters. Removing the compromise needs a distance-scaled bias in the
        // shader, which in turn needs the light position in the shadow uniform.
        const ref_dist = far_z * 0.5;
        const texel_world = 2.0 * ref_dist * @tan(fov_y * 0.5) / @as(f32, @floatFromInt(tile.size));
        d.normal_bias = @splat(s.normal_bias_texels * texel_world);

        // Position is carried for completeness; kind stays 0 because a spot is
        // a single map and needs no face selection.
        d.pos_kind = .{ l.position.x, l.position.y, l.position.z, 0 };
        // No distance fade: the cone's own attenuation reaches zero at range.
        d.fade = .{ 0, 0, 0, 0 };

        self.casters.append(self.allocator, .{
            .view_proj = view_proj,
            .rect = tile.rect,
            .tile_x = tile.x,
            .tile_y = tile.y,
            .tile_size = tile.size,
            .depth_bias = s.depth_bias,
            .depth_bias_slope = s.slope_bias,
            .cull = s.caster_cull,
            .frustum = Frustum.fromViewProj(view_proj),
        }) catch {};
        // Parallel to casters: true means the depth pass skips this view because
        // what is already in the tile is still correct.
        self.cached_view.append(self.allocator, reuse) catch {};
        // Marked valid now, since the depth pass below will fill it this frame.
        if (entry) |e| e.valid = true;

        self.data[self.data_count] = d;
        self.data_count += 1;
        return true;
    }

    /// The six cube-face axes, in the order the shader's cubeFace() reports:
    /// +X, -X, +Y, -Y, +Z, -Z.
    ///
    /// The up vectors only have to be non-degenerate against their axis. Real
    /// cubemaps fix them to match a hardware sampling convention, but nothing
    /// here samples a cubemap: each face is an ordinary 2D tile in the atlas and
    /// the shader applies whatever matrix this produced. The choice cancels out,
    /// as long as build and sample agree -- and they do, because sampling never
    /// sees it.
    const cube_faces = [6]struct { dir: Vec3, up: Vec3 }{
        .{ .dir = .{ .x = 1, .y = 0, .z = 0 }, .up = .{ .x = 0, .y = 1, .z = 0 } },
        .{ .dir = .{ .x = -1, .y = 0, .z = 0 }, .up = .{ .x = 0, .y = 1, .z = 0 } },
        .{ .dir = .{ .x = 0, .y = 1, .z = 0 }, .up = .{ .x = 0, .y = 0, .z = -1 } },
        .{ .dir = .{ .x = 0, .y = -1, .z = 0 }, .up = .{ .x = 0, .y = 0, .z = 1 } },
        .{ .dir = .{ .x = 0, .y = 0, .z = 1 }, .up = .{ .x = 0, .y = 1, .z = 0 } },
        .{ .dir = .{ .x = 0, .y = 0, .z = -1 }, .up = .{ .x = 0, .y = 1, .z = 0 } },
    };

    /// Point lights shadow in every direction, so they get six 90-degree
    /// perspective views arranged as a cube around the light. Six tiles and six
    /// depth passes make this by far the most expensive light to shadow -- the
    /// same budget as six spots -- which is why engines shadow only a handful of
    /// point lights and leave the rest unshadowed.
    ///
    /// The faces are stored as six independent atlas tiles rather than a real
    /// cubemap. That keeps one atlas, one sampler and one code path for all
    /// three light types, at the cost of filtering across face seams: hardware
    /// cube sampling would interpolate between faces, whereas PCF taps here are
    /// clamped inside their tile (see sampleShadow) and repeat the edge texel.
    fn buildPoint(self: *ShadowRenderer, l: Light, s: ShadowSettings, camera: Camera3D) bool {
        // Same reach test as a spot, and it matters six times as much here.
        const to_light = l.position.sub(camera.position).length();
        if (to_light - l.range > s.max_distance) return false;

        const near_z = @max(0.05, l.range * 0.01);
        const far_z = @max(near_z + 0.1, l.range);
        // 90 degrees exactly: six of them tile a full sphere with no gap and no
        // overlap, which is the whole reason a cube is used rather than some
        // other arrangement of frusta.
        const fov_y = std.math.degreesToRadians(90.0);
        const light_proj = zm.perspectiveFovLh(fov_y, 1.0, near_z, far_z);

        var d: ShadowData = std.mem.zeroes(ShadowData);
        d.params = .{
            6.0, // six views
            @floatFromInt(s.pcf_radius),
            0.0, // nothing to blend between; faces are adjacent, not nested
            1.0 / @as(f32, @floatFromInt(self.atlas.size)),
        };
        d.splits = @splat(far_z);
        d.pos_kind = .{ l.position.x, l.position.y, l.position.z, 1.0 };
        // No distance fade: inverse-square falloff has already taken this light
        // to nothing by the time its shadow range ends.
        d.fade = .{ 0, 0, 0, 0 };

        // Every face shares the fov and range, so one bias serves all six.
        var texel_world: f32 = 0;

        var f: usize = 0;
        while (f < cube_faces.len) : (f += 1) {
            // Partial allocation would leave a light shadowing correctly in some
            // directions and not others, which reads as geometry randomly not
            // casting. Better to drop the light: bail and let it go unshadowed.
            const tile = self.atlas.allocate(s.resolution) orelse return false;

            const face = cube_faces[f];
            const light_view = zm.lookAtLh(
                v4(l.position, 1),
                v4(l.position.add(face.dir), 1),
                v4(face.up, 0),
            );
            const view_proj = zm.mul(light_view, light_proj);

            d.view_proj[f] = matToArr(view_proj);
            d.rect[f] = tile.rect.uv;

            if (f == 0) {
                const ref_dist = far_z * 0.5;
                texel_world = 2.0 * ref_dist * @tan(fov_y * 0.5) / @as(f32, @floatFromInt(tile.size));
            }

            self.casters.append(self.allocator, .{
                .view_proj = view_proj,
                .rect = tile.rect,
                .tile_x = tile.x,
                .tile_y = tile.y,
                .tile_size = tile.size,
                .depth_bias = s.depth_bias,
                .depth_bias_slope = s.slope_bias,
                .cull = s.caster_cull,
                .frustum = Frustum.fromViewProj(view_proj),
            }) catch {};
            // Point lights are not cached yet, but the flag array has to stay
            // index-aligned with casters or the skips below would apply to the
            // wrong views.
            self.cached_view.append(self.allocator, false) catch {};
        }

        d.normal_bias = @splat(s.normal_bias_texels * texel_world);

        self.data[self.data_count] = d;
        self.data_count += 1;
        return true;
    }

    // -----------------------------------------------------------------------
    //  Render the depth-only passes
    // -----------------------------------------------------------------------

    /// Render every caster into the atlas. Call once per frame, after build(),
    /// BEFORE the main geometry pass. `draw_fn` is a user callback that issues the
    /// shadow-casting geometry for the current caster's view_proj — this keeps
    /// ShadowRenderer decoupled from how the scene stores its renderables.
    pub fn render(self: *ShadowRenderer, scene: anytype) void {
        if (self.casters.items.len == 0) return;

        // 1) Clear.
        //
        // The whole-atlas clear is the fast path and stays the default, but it
        // would wipe every cached tile along with everything else. When any tile
        // holds valid cached depth, each view that IS being redrawn clears its
        // own region instead, by way of the viewport already applied below: the
        // pass loads existing depth, and the caster draws overwrite the tile's
        // contents wherever geometry covers it.
        //
        // Tiles being redrawn therefore need their stale depth cleared first,
        // which is what clearTileDepth does.
        if (!self.atlas.has_cached_tiles) {
            zupra.beginDrawingPass(self.atlas.clearPass());
            zupra.endDrawing();
        }

        // 2) Each caster renders into its tile (LOAD, viewport-confined).
        //
        // The origin flag matters and is easy to get wrong. Tile coordinates
        // come out of the allocator measured top-down (row 0 first), and the
        // atlas rect handed to the shader is derived from those same
        // coordinates. Passing a fixed `false` here asks sokol to interpret
        // tile_y bottom-up, which on D3D11/Metal/WebGPU mirrors every tile to
        // the opposite end of the atlas: the depth pass writes one region and
        // the lighting pass samples another, so every comparison hits cleared
        // depth (1.0), passes, and the scene renders fully lit with no shadow
        // anywhere. Using the backend's own origin convention makes tile_y
        // address the same texel rows that a texture fetch of the rect returns,
        // on every backend.
        const origin_top_left = sg.queryFeatures().origin_top_left;

        const sig = self.depthPassSignature();
        for (self.casters.items, 0..) |caster, i| {
            // The whole point of the cache: a view whose depth is still correct
            // costs nothing at all, no clear and no geometry.
            if (i < self.cached_view.items.len and self.cached_view.items[i]) continue;

            zupra.beginDrawingPass(self.atlas.casterPass());
            sg.applyViewport(
                @intCast(caster.tile_x),
                @intCast(caster.tile_y),
                @intCast(caster.tile_size),
                @intCast(caster.tile_size),
                origin_top_left,
            );
            sg.applyScissorRect(
                @intCast(caster.tile_x),
                @intCast(caster.tile_y),
                @intCast(caster.tile_size),
                @intCast(caster.tile_size),
                origin_top_left,
            );
            // The scene draws its shadow casters through us for this view_proj.
            scene.drawShadowCasters(self, caster.view_proj, caster.frustum, caster.depth_bias, caster.depth_bias_slope, caster.cull, sig);
            zupra.endDrawing();
        }
    }

    /// Draw one mesh into the current caster's depth tile. Called by the scene's
    /// drawShadowCasters callback for each casting instance. Position-only — no
    /// material, no fragment work beyond depth.
    pub fn drawMesh(self: *ShadowRenderer, mesh: Mesh, model: Matrix, view_proj: Matrix, depth_bias: f32, slope_bias: f32, cull: sg.CullMode, sig: PassSignature) void {
        var key = PipelineKey{
            .shader = self.shader,
            .layout = .mesh, // reuse mesh layout; only pos is consumed
            .index_type = if (mesh.index_type == .u16) .u16 else .u32,
            .indexed = true,
            .pass = sig,
            .primitive = .TRIANGLES,
            .cull = cull, // see ShadowSettings.caster_cull
            .blend = .none,
            .depth_test = true,
            .depth_write = true,
        };
        // Rasterizer-stage bias, applied while the map is written rather than
        // when it's sampled. Doing it here means the stored depth is already
        // conservative for every consumer of the tile, and it costs nothing --
        // the alternative, biasing the comparison in the shader, has to be
        // repeated in every path that samples a shadow.
        key.setDepthBias(depth_bias, slope_bias, 0.0);

        const pip = self.cache.get(key) catch return;

        var vs = shd_depth.VsParams{
            .mvp = matToArr(zm.mul(model, view_proj)),
        };

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = mesh.vbuf;
        bindings.index_buffer = mesh.ibuf;

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(shd_depth.UB_vs_params, sg.asRange(&vs));
        sg.draw(0, mesh.index_count, 1);
    }

    /// Draw one mesh once per supplied model matrix, as a single instanced draw.
    ///
    /// `models` are row-major matrices in the engine's usual layout; they go to
    /// the GPU untouched and the shader reconstructs them (see the note in
    /// shadow_depth_instanced.glsl about rows becoming columns).
    ///
    /// The batch key upstream is the MESH, not the material -- a depth pass has
    /// no material, so every caster sharing geometry collapses into one call
    /// regardless of how it looks.
    pub fn drawMeshInstanced(
        self: *ShadowRenderer,
        mesh: Mesh,
        models: []const Matrix,
        view_proj: Matrix,
        depth_bias: f32,
        slope_bias: f32,
        cull: sg.CullMode,
        sig: PassSignature,
    ) void {
        if (models.len == 0) return;

        var key = PipelineKey{
            .shader = self.inst_shader,
            .layout = .mesh_instanced,
            .index_type = mesh.index_type,
            .indexed = true,
            .pass = sig,
            .primitive = .TRIANGLES,
            .cull = cull, // see ShadowSettings.caster_cull
            .blend = .none,
            .depth_test = true,
            .depth_write = true,
        };
        key.setDepthBias(depth_bias, slope_bias, 0.0);

        const pip = self.cache.get(key) catch return;

        // appendBuffer sub-allocates within the frame's single stream buffer and
        // hands back the byte offset this batch landed at. Many appends per
        // frame are fine; many updateBuffer calls would not be.
        const offset = sg.appendBuffer(self.inst_buf, sg.asRange(models));
        if (sg.queryBufferOverflow(self.inst_buf)) {
            // Overflow drops the append silently, and the draw would then read
            // stale matrices -- geometry appearing at last frame's positions in
            // the shadow map, which is hard to recognise for what it is.
            std.log.err(
                "ShadowRenderer: instance buffer overflow ({d} bytes capacity) — raise max_instance_bytes",
                .{max_instance_bytes},
            );
            return;
        }

        var vs = shd_depth_inst.VsParams{ .view_proj = matToArr(view_proj) };

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = mesh.vbuf;
        bindings.vertex_buffers[1] = self.inst_buf;
        bindings.vertex_buffer_offsets[1] = offset;
        bindings.index_buffer = mesh.ibuf;

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(shd_depth_inst.UB_vs_params, sg.asRange(&vs));
        sg.draw(0, mesh.index_count, @intCast(models.len));
    }

    pub fn drawModel(self: *ShadowRenderer, inst: ModelInstance, view_proj: Matrix, depth_bias: f32, slope_bias: f32, sig: PassSignature) void {
        const m = inst.modelMatrix();
        for (inst.model.meshes) |submesh| {
            self.drawMesh(submesh, m, view_proj, depth_bias, slope_bias, sig);
        }
    }

    // -----------------------------------------------------------------------
    //  Bind for the lighting pass
    // -----------------------------------------------------------------------

    pub fn atlasView(self: ShadowRenderer) sg.View {
        return self.atlas.sample_view;
    }
    pub fn compareSampler(self: ShadowRenderer) sg.Sampler {
        return self.atlas.compare_sampler;
    }
    pub fn shadowData(self: *const ShadowRenderer) []const ShadowData {
        return self.data[0..self.data_count];
    }
};

// ---------------------------------------------------------------------------
//  Directional cascade fitting
// ---------------------------------------------------------------------------

/// Fit an orthographic light-space view-projection to the camera frustum slice
/// [near_split, far_split]. STABLE: quantises the volume to whole shadow texels
/// so the map samples the same world points frame to frame, which is what stops
/// shadow edges from crawling and boiling as the camera moves. Getting this
/// right is most of the difference between a real CSM and a naive one.
/// A fitted cascade: its light-space view-projection, plus how much world space
/// one of its shadow texels covers. The latter is what lets bias be specified
/// once in texels and stay correct across cascades of very different extents.
const CascadeFit = struct {
    view_proj: Matrix,
    /// World units per shadow texel in this cascade.
    texel_world: f32,
};

fn fitDirectionalCascade(
    camera: Camera3D,
    light_dir: Vec3,
    near_split: f32,
    far_split: f32,
    tile_size: u32,
    caster_extrusion: f32,
) CascadeFit {
    // Frustum-slice corners in world space.
    var corners: [8]Vec3 = undefined;
    frustumSliceCorners(camera, near_split, far_split, &corners);

    // Bounding SPHERE of the slice. A sphere rather than an AABB because its
    // radius depends only on the camera's field of view and the split
    // distances, never on where the camera is or which way it points. The
    // volume therefore keeps a constant size as the camera moves, so a fixed
    // world-units-per-texel ratio holds and snapping is meaningful.
    var center = Vec3{ .x = 0, .y = 0, .z = 0 };
    for (corners) |c| center = center.add(c);
    center = center.mul(1.0 / 8.0);

    var radius: f32 = 0;
    for (corners) |c| radius = @max(radius, c.sub(center).length());
    // Quantise the radius too: floating-point drift in the corner maths would
    // otherwise jitter it slightly every frame and undo the snap below.
    radius = @ceil(radius * 16.0) / 16.0;

    // Light-space basis anchored at the WORLD ORIGIN rather than at the slice
    // centre. This matters: a view matrix built looking at the centre places
    // that centre at its own origin by construction, so quantising it there is
    // a no-op and the volume still slides continuously with the camera. Fixing
    // the basis to the origin gives the snap a stationary grid to snap to.
    const up = pickUp(light_dir);
    const light_view = zm.lookAtLh(zm.f32x4(0, 0, 0, 1), v4(light_dir, 1), v4(up, 0));

    var center_ls = zm.mul(v4(center, 1.0), light_view);

    // TEXEL SNAP: move the volume in whole-texel steps only. Sub-texel motion
    // is what makes an edge shimmer, because each frame the same world point
    // lands on a different part of a texel and flips its depth comparison.
    const texels_per_unit = @as(f32, @floatFromInt(tile_size)) / (radius * 2.0);
    center_ls[0] = @floor(center_ls[0] * texels_per_unit) / texels_per_unit;
    center_ls[1] = @floor(center_ls[1] * texels_per_unit) / texels_per_unit;

    const l = center_ls[0] - radius;
    const r = center_ls[0] + radius;
    const b = center_ls[1] - radius;
    const t = center_ls[1] + radius;

    // Depth range along the light. The near plane reaches back past the volume
    // by caster_extrusion so casters standing between the light and the visible
    // slice are still rendered; the far plane only needs to reach the back of
    // the sphere.
    const near_z = center_ls[2] - radius - caster_extrusion;
    const far_z = center_ls[2] + radius;
    const light_proj = zm.orthographicOffCenterLh(l, r, b, t, near_z, far_z);

    return .{
        .view_proj = zm.mul(light_view, light_proj),
        // The box spans 2*radius across tile_size texels. Same quantity the snap
        // above works in, so the two can't disagree.
        .texel_world = (radius * 2.0) / @as(f32, @floatFromInt(tile_size)),
    };
}

/// The 8 world-space corners of the camera frustum between two view distances.
fn frustumSliceCorners(camera: Camera3D, near_d: f32, far_d: f32, out: *[8]Vec3) void {
    const fwd = camera.forward();
    const right = fwd.cross(camera.up).normalize();
    const up = right.cross(fwd).normalize();

    const tan_v = @tan(camera.fov_y * 0.5);
    const tan_h = tan_v * camera.aspect;

    inline for (.{ near_d, far_d }, 0..) |dist, di| {
        const cx = camera.position.add(fwd.mul(dist));
        const hh = tan_v * dist;
        const hw = tan_h * dist;
        const ru = right.mul(hw);
        const uu = up.mul(hh);
        out[di * 4 + 0] = cx.add(uu).add(ru); // top-right
        out[di * 4 + 1] = cx.add(uu).sub(ru); // top-left
        out[di * 4 + 2] = cx.sub(uu).add(ru); // bottom-right
        out[di * 4 + 3] = cx.sub(uu).sub(ru); // bottom-left
    }
}

/// Practical split scheme: blend of logarithmic and uniform (Zhang et al.).
/// lambda=0.5 balances near-detail against far-coverage.
fn computeSplits(near: f32, far: f32, count: u32, out: *[max_cascades + 1]f32) void {
    const lambda: f32 = 0.7;
    const n: f32 = @max(near, 0.05);
    out[0] = n;
    var i: u32 = 1;
    while (i <= count) : (i += 1) {
        const p = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(count));
        const log_split = n * std.math.pow(f32, far / n, p);
        const uni_split = n + (far - n) * p;
        out[i] = lambda * log_split + (1.0 - lambda) * uni_split;
    }
    // Fill any unused entries with far so cascade selection is well-defined.
    while (i <= max_cascades) : (i += 1) out[i] = far;
}

/// Choose an up vector not parallel to the light direction.
fn vecEql(a: Vec3, b: Vec3) bool {
    return a.x == b.x and a.y == b.y and a.z == b.z;
}

fn pickUp(dir: Vec3) Vec3 {
    if (@abs(dir.y) > 0.99) return .{ .x = 0, .y = 0, .z = 1 };
    return .{ .x = 0, .y = 1, .z = 0 };
}

fn v4(p: Vec3, w: f32) zm.Vec {
    return zm.f32x4(p.x, p.y, p.z, w);
}

fn matToArr(m: Matrix) [16]f32 {
    return @bitCast(m);
}
