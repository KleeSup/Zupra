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

const Matrix = math.Matrix;
const Vec3 = math.Vec3;
const Light = light_mod.Light;
const ShadowAtlas = shadow.ShadowAtlas;
const ShadowCaster = shadow.ShadowCaster;
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
/// Max simultaneously shadowed lights (any type). The shadow_data shader array
/// is sized to this. Generous, but each entry is small.
pub const max_shadowed = 16;

/// One shadowed light's data, uploaded parallel to the light list. The shader
/// indexes this by the light's shadow_index (stored in the light texture).
pub const ShadowData = extern struct {
    /// Light-space view-projection. For cascades, this is per-cascade (see
    /// cascade_vp). For a single map, only [0] is used.
    view_proj: [max_cascades][16]f32,
    /// Atlas rect per cascade (xy origin, zw size, in [0,1] UV).
    rect: [max_cascades][4]f32,
    /// x = cascade count, y = pcf radius (texels), z = normal bias (world),
    /// w = atlas texel size (1/atlas_size, for PCF tap spacing).
    params: [4]f32,
    /// Cascade split distances (view-space), x..w = splits 0..3 far planes.
    /// Used to pick the cascade for a fragment by its depth.
    splits: [4]f32,
};

pub const ShadowRenderer = struct {
    cache: *PipelineCache,
    atlas: ShadowAtlas,
    shader: sg.Shader,

    /// Casters built this frame (all lights, all cascades/faces).
    casters: std.ArrayListUnmanaged(ShadowCaster) = .empty,
    /// Per-shadowed-light data, uploaded to the lighting shader.
    data: [max_shadowed]ShadowData = undefined,
    data_count: u32 = 0,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, cache: *PipelineCache, opts: shadow.AtlasOptions) ShadowRenderer {
        const shader = sg.makeShader(shd_depth.shadowDepthShaderDesc(sg.queryBackend()));
        // Diagnostic: if the depth-only shader failed to compile, every shadow
        // draw will die at apply_uniforms with "pipeline no longer alive". Catch
        // it here where the cause is obvious.
        const state = sg.queryShaderState(shader);
        if (state != .VALID) {
            std.log.err("ShadowRenderer: shadow_depth shader is {s} (not VALID) — depth pass will fail", .{@tagName(state)});
        }
        return .{
            .cache = cache,
            .atlas = ShadowAtlas.init(opts),
            .shader = shader,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ShadowRenderer) void {
        self.casters.deinit(self.allocator);
        self.atlas.deinit();
        sg.destroyShader(self.shader);
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
    pub fn build(self: *ShadowRenderer, lights: []Light, camera: Camera3D) void {
        self.casters.clearRetainingCapacity();
        self.data_count = 0;
        self.atlas.reset();

        for (lights) |*l| {
            l.shadow_index = -1; // default: unshadowed this frame
            if (!l.shadow.enabled) continue;
            if (self.data_count >= max_shadowed) continue; // out of slots -> unshadowed

            const assigned: bool = switch (l.type) {
                .directional => self.buildDirectional(l.*, l.shadow, camera),
                // .spot => self.buildSpot(l.*, l.shadow, camera),   // Phase 3
                // .point => self.buildPoint(l.*, l.shadow, camera), // Phase 3
                else => false,
            };
            if (assigned) {
                l.shadow_index = @intCast(self.data_count - 1);
            }
        }
    }

    fn buildDirectional(self: *ShadowRenderer, l: Light, s: ShadowSettings, camera: Camera3D) bool {
        const cascades = std.math.clamp(s.cascade_count, 1, max_cascades);
        var d: ShadowData = std.mem.zeroes(ShadowData);
        d.params = .{
            @floatFromInt(cascades),
            @floatFromInt(s.pcf_radius),
            s.normal_bias,
            1.0 / @as(f32, @floatFromInt(self.atlas.size)),
        };

        // Cascade split distances (practical logarithmic split): near..max_distance
        // partitioned so nearer cascades (which cover fewer world units per texel)
        // get proportionally more of the range. Single cascade = the whole range.
        var splits: [max_cascades + 1]f32 = undefined;
        computeSplits(camera.near, s.max_distance, cascades, &splits);
        d.splits = .{ splits[1], splits[2], splits[3], splits[4] };

        const light_dir = l.direction.normalize();

        var c: u32 = 0;
        while (c < cascades) : (c += 1) {
            const tile = self.atlas.allocate(s.resolution) orelse break; // atlas full
            const vp = fitDirectionalCascade(camera, light_dir, splits[c], splits[c + 1], tile.size, s.caster_extrusion);

            d.view_proj[c] = matToArr(vp);
            d.rect[c] = tile.rect.uv;

            self.casters.append(self.allocator, .{
                .view_proj = vp,
                .rect = tile.rect,
                .tile_x = tile.x,
                .tile_y = tile.y,
                .tile_size = tile.size,
                .depth_bias = s.depth_bias,
                .depth_bias_slope = s.slope_bias,
            }) catch {};
        }

        // If the atlas was full before even cascade 0 got a tile, this light is
        // unshadowed this frame (rect stays zero-size).
        if (d.rect[0][2] <= 0.0) return false;
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

        // 1) Clear the whole atlas depth once.
        zupra.beginDrawingPass(self.atlas.clearPass());
        zupra.endDrawing();

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
        for (self.casters.items) |caster| {
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
            scene.drawShadowCasters(self, caster.view_proj, caster.depth_bias, caster.depth_bias_slope, sig);
            zupra.endDrawing();
        }
    }

    /// Draw one mesh into the current caster's depth tile. Called by the scene's
    /// drawShadowCasters callback for each casting instance. Position-only — no
    /// material, no fragment work beyond depth.
    pub fn drawMesh(self: *ShadowRenderer, mesh: Mesh, model: Matrix, view_proj: Matrix, depth_bias: f32, slope_bias: f32, sig: PassSignature) void {
        var key = PipelineKey{
            .shader = self.shader,
            .layout = .mesh, // reuse mesh layout; only pos is consumed
            .index_type = if (mesh.index_type == .u16) .u16 else .u32,
            .indexed = true,
            .pass = sig,
            .primitive = .TRIANGLES,
            .cull = .FRONT, // front-face cull reduces peter-panning / acne
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
fn fitDirectionalCascade(
    camera: Camera3D,
    light_dir: Vec3,
    near_split: f32,
    far_split: f32,
    tile_size: u32,
    caster_extrusion: f32,
) Matrix {
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

    return zm.mul(light_view, light_proj);
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
