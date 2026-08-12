//! src/render/scene.zig
//!
//! SceneRenderer: the high-level frame orchestrator. Pick forward OR deferred at
//! construction; the renderer then owns the right passes + targets and runs them
//! from begin/draw/end. The user submits draws and gets a correct frame without
//! wiring sokol passes by hand.
//!
//!   * Forward mode:  opaque -> scene-color(HDR) -> [transparent] -> present.
//!     Cheap, low VRAM (no G-buffer), full PBR with few lights. Mobile/wasm.
//!   * Deferred mode: geometry -> G-buffer -> lighting -> scene-color(HDR) ->
//!     [transparent] -> present. Many lights; unlocks screen-space effects.
//!
//! Both render LINEAR HDR into the scene-color target and finish with the shared
//! Present (tonemap) pass — the common substrate.
//!
//! Routing: each submesh is routed by material.alpha_mode. opaque_/mask go to
//! the mode's opaque path; blend is collected into the transparent queue. The
//! sorted transparent forward pass is the NEXT step — entries are collected now
//! (real structure) but not yet drawn.

const std = @import("std");
const sg = @import("sokol").gfx;
const sapp = @import("sokol").app;
const zupra = @import("../root.zig");

const pipeline = @import("../graphics/pipeline.zig");
const math = @import("../math.zig");
const Camera3D = @import("camera3d.zig").Camera3D;
const Environment = @import("environment.zig").Environment;
const mesh_mod = @import("mesh.zig");
const model_mod = @import("model.zig");
const material_mod = @import("material.zig");
const gbuffer_mod = @import("gbuffer.zig");
const deferred_mod = @import("deferred.zig");
const Framebuffer = @import("framebuffer.zig").Framebuffer;
const Present = @import("present.zig").Present;

const PipelineCache = pipeline.PipelineCache;
const PassSignature = pipeline.PassSignature;
const Mesh = mesh_mod.Mesh;
const MeshRenderer = mesh_mod.MeshRenderer;
const DepthPrepass = @import("depth_prepass.zig").DepthPrepass;
const culling = @import("culling.zig");
const Sphere = culling.Sphere;
const Frustum = culling.Frustum;
const Material = material_mod.Material;
const ModelInstance = model_mod.ModelInstance;
const GBuffer = gbuffer_mod.GBuffer;
const GeometryRenderer = deferred_mod.GeometryRenderer;
const DeferredRenderer = deferred_mod.DeferredRenderer;
const Matrix = math.Matrix;
const Color = zupra.Color;
const Skybox = @import("skybox.zig").Skybox;
const Ibl = @import("ibl.zig").Ibl;
const PostChain = @import("posprocess.zig").PostChain;
const AAMethod = @import("posprocess.zig").AAMethod;
const EnvironmentMap = @import("render.zig").EnvironmentMap;
const Ssao = @import("render.zig").Ssao;
const Bloom = @import("render.zig").Bloom;
const Taa = @import("taa.zig").Taa;
const VelocityPass = @import("velocity.zig").VelocityPass;

const ShadowRenderer = @import("shadow_renderer.zig").ShadowRenderer;
const ShadowParams = @import("shaders").mesh.ShadowParams;
const shadow_max_views = @import("shadow_renderer.zig").max_shadow_views;

pub const ShadingMode = enum { forward, deferred };

const TransparentEntry = struct {
    mesh: Mesh,
    model: Matrix,
    material: Material,
    depth: f32, // squared distance to camera, for back-to-front sort
    bounds: Sphere,
};

/// One submesh queued for the shadow passes, with its world-space bounds fitted
/// once at submission.
///
/// Flattened to submeshes rather than kept as instances so the list can be
/// sorted into contiguous runs of identical geometry, which is what lets each
/// run become a single instanced draw. Fitting the bounds here also matters:
/// the caster loop walks this list once per cascade and per cube face, so a
/// bound computed in there would be recomputed a dozen times a frame for an
/// object that never moved.
const ShadowSubmission = struct {
    mesh: Mesh,
    model: Matrix,
    bounds: Sphere,
    /// Sort key. The vertex buffer id identifies the geometry, and geometry is
    /// the whole batch key here -- a depth pass has no material, so twelve
    /// differently coloured spheres sharing a mesh are one draw.
    key: u32,
    cacheable: bool,
};

/// A submesh queued for the main opaque pass, with the world bounds camera
/// culling tests against.
const OpaqueEntry = struct {
    mesh: Mesh,
    model: Matrix,
    material: Material,
    bounds: Sphere,
    /// Squared distance to the camera, for front-to-back ordering. Already
    /// computed for the transparent sort, so carrying it costs nothing.
    depth: f32,
};

/// A submesh deferred mode couldn't put in the G-buffer (custom shader, or a
/// shading model the G-buffer layout can't express) and has to forward-shade
/// after the lighting pass.
///
/// Deliberately without bounds: entries only get here via drawOpaque, which
/// runs downstream of the camera test, so re-testing would be redundant work
/// and a second place for the two results to disagree.
const ForwardEntry = struct {
    mesh: Mesh,
    model: Matrix,
    material: Material,
};

const MovedEntry = struct {
    mesh: Mesh,
    model: Matrix,
    prev_model: Matrix,
    material: Material,
};

pub const SceneRenderer = struct {
    mode: ShadingMode,
    cache: *PipelineCache,
    allocator: std.mem.Allocator,
    width: u32,
    height: u32,
    clear_color: Color = .{ .r = 0.02, .g = 0.02, .b = 0.03, .a = 1 },
    render_scale: u8 = 1,
    msaa_samples: u8 = 1,

    scene_color: Framebuffer,
    post: PostChain,
    skybox: ?Skybox = null,
    ibl: ?Ibl = null,
    ibl_baked: bool = false,

    /// Captured environment. When set it replaces the procedural sky, both as
    /// the visible background and as the source the IBL chain bakes from.
    /// SceneRenderer doesn't own it, an environment map is commonly shared between scenes and is expensive to duplicate.
    envmap: ?*EnvironmentMap = null,

    shadows: ShadowRenderer,
    shadow_params: ShadowParams = undefined,
    shadow_queue: std.ArrayList(ShadowSubmission) = .empty,

    // deferred-only
    gbuffer: GBuffer = undefined,
    geo: GeometryRenderer = undefined,
    lit: DeferredRenderer = undefined,

    // forward-only
    forward: MeshRenderer = undefined,
    prepass: DepthPrepass = undefined,

    /// Lay depth down before shading in forward mode. Costs one position-only
    /// draw per eligible submesh and saves shading every fragment that ends up
    /// hidden -- worth it whenever geometry overlaps in screen space, which is
    /// most scenes. The deferred path ignores this: its G-buffer pass already
    /// resolves visibility before any lighting runs.
    depth_prepass: bool = true,
    prepass_fb: Framebuffer = undefined,

    /// Depth DRAW CALLS issued by the shadow passes last frame, after culling
    /// and batching, and the number of instances those draws covered. The gap
    /// between them is what instancing is buying; the gap between instances and
    /// queue length times caster count is what culling is buying.
    shadow_draws: u32 = 0,
    shadow_instances: u32 = 0,

    ssao: Ssao = undefined,
    /// Screen-space occlusion. Deferred only: it needs the G-buffer's normals,
    /// which the forward path does not produce.
    ssao_enabled: bool = true,

    /// Temporal anti-aliasing. Accumulates sub-pixel-jittered frames, which
    /// resolves edges far beyond what FXAA reaches and averages out per-pixel
    /// noise in stochastic effects. Costs one full-screen pass and two RGBA16F
    /// history targets.
    taa: Taa = undefined,
    velocity: VelocityPass = undefined,

    bloom: Bloom = undefined,
    /// HDR bloom. Costs roughly a dozen small fullscreen passes, although the widest
    /// levels run on tiny images, so the total is far less than it looks.
    bloom_enabled: bool = true,

    /// Submeshes submitted vs. those that survived camera culling last frame.
    submitted_draws: u32 = 0,
    visible_draws: u32 = 0,
    /// Instances drawn into shadow tiles whose geometry did not move this frame.
    /// These are what caching would eliminate. Compare against shadow_instances
    /// for the fraction of shadow work that is redundant.
    shadow_static_instances: u32 = 0,

    /// Skip camera frustum culling. Useful when debugging: if geometry vanishes
    /// unexpectedly, turning this off says immediately whether culling is the
    /// cause or something downstream is.
    frustum_culling: bool = true,

    /// This frame's camera frustum, rebuilt in end() once the camera is final.
    camera_frustum: Frustum = .{},

    /// Reused per-batch matrix scratch. Held on the renderer rather than built
    /// per draw so the allocation happens once and then never again.
    instance_scratch: std.ArrayList(Matrix) = .empty,
    forward_opaque: std.ArrayList(ForwardEntry) = .empty,

    // Opaque submissions for this frame. draw() records into this; end() replays
    // it into whichever pass the mode calls for. Recording rather than drawing
    // immediately is what lets the shadow depth passes -- which must run BEFORE
    // the main geometry pass, yet need to know the frame's geometry -- see the
    // whole scene.
    opaque_queue: std.ArrayList(OpaqueEntry) = .empty,

    // transparent draw queue (both modes); flushed sorted in the NEXT step
    transparent: std.ArrayList(TransparentEntry) = .empty,

    moved_queue: std.ArrayList(MovedEntry) = .empty,

    // per-frame captured state
    camera: Camera3D = undefined,
    env: *Environment = undefined,

    pub fn init(allocator: std.mem.Allocator, cache: *PipelineCache, mode: ShadingMode, width: u32, height: u32) SceneRenderer {
        var self = SceneRenderer{
            .mode = mode,
            .cache = cache,
            .allocator = allocator,
            .width = width,
            .height = height,
            .scene_color = undefined,
            .post = undefined,
            .shadows = ShadowRenderer.init(allocator, cache, .{}),
        };
        self.scene_color = self.makeSceneColor(mode, width, height);
        self.post = PostChain.init(allocator, cache, width, height, self.clear_color);
        // Forward renderer exists in BOTH modes: forward mode uses it for
        // opaque, and both modes use it to forward-shade transparents.
        self.forward = MeshRenderer.init(cache);
        self.prepass = DepthPrepass.init(cache);
        self.ssao = Ssao.init(cache, width, height, .{});
        if (mode == .deferred) {
            self.gbuffer = GBuffer.init(width, height);
            self.geo = GeometryRenderer.init(cache);
            self.lit = DeferredRenderer.init(cache);
        } else if (mode == .forward) {
            self.prepass_fb = Framebuffer.init(.{
                .width = width,
                .height = height,
                .color_format = .RGBA16F,
                .depth_format = .DEPTH,
            });
        }
        self.skybox = Skybox.init(cache);
        self.ibl = Ibl.init(allocator, cache);
        self.bloom = Bloom.init(cache, width, height, .{});
        self.taa = Taa.init(cache, width, height, .{});
        self.velocity = VelocityPass.init(cache, width, height);
        return self;
    }

    pub fn deinit(self: *SceneRenderer) void {
        self.transparent.deinit(self.allocator);
        self.forward_opaque.deinit(self.allocator);
        self.opaque_queue.deinit(self.allocator);
        self.instance_scratch.deinit(self.allocator);
        self.moved_queue.deinit(self.allocator);
        if (self.mode == .deferred) {
            self.lit.deinit();
            self.gbuffer.deinit();
            self.geo.deinit();
        } else if (self.mode == .forward) {
            self.prepass_fb.deinit();
        }
        self.ssao.deinit();
        self.forward.deinit();
        self.prepass.deinit();
        self.post.deinit();
        self.scene_color.deinit();
        if (self.skybox) |*s| s.deinit();
        if (self.ibl) |*ibl| ibl.deinit();
        self.shadows.deinit();
        self.shadow_queue.deinit(self.allocator);
        self.bloom.deinit();
        self.taa.deinit();
        self.velocity.deinit();
    }

    pub fn setRenderScale(self: *SceneRenderer, scale: u8) void {
        if (scale == self.render_scale) return;
        self.render_scale = @max(1, scale);
        self.width = 0; // force ensureSize to rebuild next frame
    }

    pub fn setMsaa(self: *SceneRenderer, samples: u8) void {
        if (samples == self.msaa_samples) return;
        self.msaa_samples = @max(1, samples);
        self.width = 0; // force ensureSize to recreate the target next frame
    }

    /// Swap the environment. Clears ibl_baked so irradiance and the specular
    /// prefilter are rebuilt from the new source on the next frame -> leaving the
    /// old bake in place would light the scene from an environment no longer visible behind it.
    pub fn setEnvironmentMap(self: *SceneRenderer, map: ?*EnvironmentMap) void {
        self.envmap = map;
        self.ibl_baked = false;
    }

    fn makeSceneColor(self: *SceneRenderer, mode: ShadingMode, w: u32, h: u32) Framebuffer {
        return Framebuffer.init(.{
            .width = w,
            .height = h,
            .color_format = .RGBA16F,
            // forward writes opaque depth here; deferred lighting is fullscreen
            // (its depth lives in the G-buffer), so none needed yet.
            .depth_format = if (mode == .forward) .DEPTH else .NONE,
            .sample_count = if (mode == .forward) self.msaa_samples else 1,
        });
    }

    fn ensureSize(self: *SceneRenderer, w: u32, h: u32) void {
        if (w == 0 or h == 0) return;
        const s: u32 = self.render_scale;
        const rw = w * s;
        const rh = h * s;
        if (rw == self.width and rh == self.height) return;
        self.scene_color.deinit();
        self.scene_color = self.makeSceneColor(self.mode, rw, rh);
        if (self.mode == .deferred) {
            self.gbuffer.deinit();
            self.gbuffer = GBuffer.init(rw, rh);
        } else if (self.mode == .forward) {
            self.prepass_fb.deinit();
            self.prepass_fb = Framebuffer.init(.{
                .width = rw,
                .height = rh,
                .color_format = .RGBA16F,
                .depth_format = .DEPTH,
            });
        }
        self.post.resize(rw, rh);
        self.ssao.resize(rw, rh);
        self.bloom.resize(rw, rh);
        self.taa.resize(rw, rh);
        self.velocity.resize(rw, rh);
        self.width = rw;
        self.height = rh;
    }

    pub fn setAAMethod(self: *SceneRenderer, method: AAMethod) void {
        if (self.post.aa == .taa and method != .taa) self.taa.resetHistory();
        self.post.aa = method;
    }

    pub fn isTaaActive(self: *SceneRenderer) bool {
        return self.post.aa == .taa;
    }

    pub fn begin(self: *SceneRenderer, camera: Camera3D, env: *Environment) void {
        const w: u32 = @intFromFloat(sapp.widthf());
        const h: u32 = @intFromFloat(sapp.heightf());
        self.ensureSize(w, h);

        self.camera = camera;
        self.camera.setViewport(@floatFromInt(w), @floatFromInt(h));
        self.camera.jitter = if (self.isTaaActive()) self.taa.jitter() else .{ 0, 0 };
        self.env = env;
        self.transparent.clearRetainingCapacity();
        self.forward_opaque.clearRetainingCapacity();
        self.opaque_queue.clearRetainingCapacity();
        self.shadow_queue.clearRetainingCapacity();
        self.moved_queue.clearRetainingCapacity();
        self.shadow_draws = 0;
        self.shadow_instances = 0;
        self.submitted_draws = 0;
        self.visible_draws = 0;
        self.shadow_static_instances = 0;

        // Caster SETUP runs here rather than in end() because it stamps a
        // shadow_index onto each light, and beginFrame below bakes those indices
        // into the light texture the shading pass reads. The depth passes
        // themselves are issued from end(), once the geometry is known.
        self.shadows.build(env.lighting.store.lights.items, self.camera);

        env.lighting.beginFrame(
            camera,
            env.ambient,
            @floatFromInt(self.width),
            @floatFromInt(self.height),
        );

        if (!self.ibl_baked) {
            if (self.ibl) |*ibl| {
                if (self.skybox) |*sky| {
                    ibl.bake(sky, self.envmap);
                    self.ibl_baked = true;
                }
            }
        }
        if (self.ibl) |ibl| {
            self.forward.setIbl(ibl.irradianceView(), ibl.prefilterView(), ibl.brdfLutView(), ibl.cubeSampler());
            if (self.mode == .deferred)
                self.lit.setIbl(ibl.irradianceView(), ibl.prefilterView(), ibl.brdfLutView(), ibl.cubeSampler());
        }
    }

    /// Submit an instance for this frame. Nothing reaches the GPU here --
    /// submission only records what to draw, and end() runs the passes in
    /// dependency order.
    pub fn draw(self: *SceneRenderer, inst: ModelInstance) void {
        // One shadow submission per submesh, each with its own world bounds.
        // Per-submesh rather than per-instance so a multi-material model's parts
        // batch with matching geometry elsewhere in the scene, and so culling
        // works at the granularity that is actually drawn.
        const model_matrix = inst.modelMatrix();

        // World bounds once per submesh, reused by the shadow queue and by
        // camera culling. The shadow queue is deliberately NOT camera-culled:
        // an object behind the camera can still cast a shadow into view, so it
        // is tested against each LIGHT's frustum instead, never this one.
        if (inst.cast_shadows) {
            for (inst.model.meshes) |submesh| {
                self.shadow_queue.append(self.allocator, .{
                    .mesh = submesh,
                    .model = model_matrix,
                    .bounds = submesh.bounds.transform(model_matrix),
                    .key = submesh.vbuf.id,
                    .cacheable = inst.shadow_cacheable,
                }) catch {};
            }
        }

        const dx = self.camera.position.x - inst.position.x;
        const dy = self.camera.position.y - inst.position.y;
        const dz = self.camera.position.z - inst.position.z;
        const depth = dx * dx + dy * dy + dz * dz;

        const model = inst.model;
        for (model.meshes, 0..) |submesh, i| {
            const material = model.materials[model.mesh_material[i]];

            const bounds = submesh.bounds.transform(model_matrix);

            if (material.alpha_mode == .blend) {
                self.transparent.append(self.allocator, .{
                    .mesh = submesh,
                    .model = model_matrix,
                    .material = material,
                    .depth = depth,
                    .bounds = bounds,
                }) catch {};
            } else {
                self.opaque_queue.append(self.allocator, .{
                    .mesh = submesh,
                    .model = model_matrix,
                    .material = material,
                    .bounds = bounds,
                    .depth = depth,
                }) catch {};
            }
        }
    }

    /// Submit an instance that moved since last frame, so the velocity pass can
    /// record where it was. Only objects whose transform actually changed need
    /// this, since a static object's motion is entirely accounted for by the
    /// camera reprojection TAA already does.
    pub fn drawMoved(self: *SceneRenderer, inst: ModelInstance, prev_model: Matrix) void {
        self.draw(inst);
        const model_matrix = inst.modelMatrix();
        for (inst.model.meshes, 0..) |submesh, i| {
            self.moved_queue.append(self.allocator, .{
                .mesh = submesh,
                .model = model_matrix,
                .prev_model = prev_model,
                .material = inst.model.materials[inst.model.mesh_material[i]],
            }) catch {};
        }
    }

    /// Issue this frame's casters for one shadow view.
    ///
    /// Called once per cascade, per spot and per cube face, so without the
    /// frustum test the cost is queue length times view count -- and most of
    /// those draws are for geometry the view cannot see. A near cascade covers a
    /// few metres of a scene tens of metres across; a cube face covers a sixth
    /// of a sphere. Rejecting on a bounding sphere is one dot product per plane
    /// against a bound that was already computed at submission.
    /// Camera visibility test for one world-space bound. Conservative by
    /// design: a false accept costs one draw, a false reject loses geometry.
    fn visible(self: *const SceneRenderer, bounds: Sphere) bool {
        if (!self.frustum_culling) return true;
        // An empty bound means the mesh had no vertices to fit, which should not
        // silently disappear -- draw it and let the geometry speak for itself.
        if (bounds.isEmpty()) return true;
        return self.camera_frustum.intersectsSphere(bounds);
    }

    pub fn drawShadowCasters(
        self: *SceneRenderer,
        shadows: *ShadowRenderer,
        view_proj: Matrix,
        frustum: Frustum,
        bias: f32,
        slope: f32,
        cull: sg.CullMode,
        sig: PassSignature,
    ) void {
        const items = self.shadow_queue.items;
        var i: usize = 0;
        while (i < items.len) {
            // The queue was sorted by geometry in end(), so a run of equal keys
            // is contiguous and can be found by scanning forward.
            const key = items[i].key;
            var j = i;
            self.instance_scratch.clearRetainingCapacity();
            while (j < items.len and items[j].key == key) : (j += 1) {
                if (!frustum.intersectsSphere(items[j].bounds)) continue;
                self.instance_scratch.append(self.allocator, items[j].model) catch {};
                if (items[j].cacheable) self.shadow_static_instances += 1;
            }
            if (self.instance_scratch.items.len > 0) {
                self.shadow_draws += 1;
                self.shadow_instances += @intCast(self.instance_scratch.items.len);
                shadows.drawMeshInstanced(
                    items[i].mesh,
                    self.instance_scratch.items,
                    view_proj,
                    bias,
                    slope,
                    cull,
                    sig,
                );
            }
            i = j;
        }
    }

    fn drawOpaque(self: *SceneRenderer, mesh: Mesh, model: Matrix, material: Material) void {
        switch (self.mode) {
            .deferred => {
                if (material.shading == .pbr and !material.requiresForward()) {
                    self.geo.drawMesh(mesh, model, material); // G-buffer (PBR lighting)
                } else {
                    // Forward-shaded after the lighting pass (respects shading model,
                    // and custom shaders that don't write the G-buffer layout).
                    self.forward_opaque.append(self.allocator, .{
                        .mesh = mesh,
                        .model = model,
                        .material = material,
                    }) catch {};
                }
            },
            .forward => self.forward.draw(mesh, model, material),
        }
    }

    pub fn end(self: *SceneRenderer) void {
        // Camera frustum for this frame. Built here rather than in begin()
        // because the camera is only final once submission is done -- and the
        // shadow passes below deliberately do not use it.
        self.camera_frustum = Frustum.fromViewProj(self.camera.viewProjection());
        self.submitted_draws = @intCast(self.opaque_queue.items.len + self.transparent.items.len);

        // Opaques front to back. Nearer geometry draws first, fills the depth
        // buffer early, and everything behind it is rejected before its fragment
        // shader runs.
        //
        // This is what makes the depth prepass worth having. A prepass only
        // rejects what is ALREADY in the depth buffer, so an unsorted queue
        // rejects whatever the submission order happened to favour, and a
        // back-to-front one rejects nothing at all.
        //
        // Sorting by material would batch better, but depth wins here: the
        // pipeline cache already keys on state, and a wasted PBR evaluation
        // costs far more than a pipeline switch.
        std.mem.sort(OpaqueEntry, self.opaque_queue.items, {}, struct {
            fn lessThan(_: void, a: OpaqueEntry, b: OpaqueEntry) bool {
                return a.depth < b.depth;
            }
        }.lessThan);

        // 1) Shadow depth passes. Every caster tile is filled before anything
        // samples the atlas, and the queue they draw from is complete by now.
        //
        // Sort by geometry first, ONCE. Every caster pass then walks the same
        // ordering and finds identical meshes already adjacent, so grouping is a
        // linear scan instead of a per-view hash. Sorting here rather than in
        // draw() keeps submission cheap and costs one pass over the queue.
        std.mem.sort(ShadowSubmission, self.shadow_queue.items, {}, struct {
            fn lessThan(_: void, a: ShadowSubmission, b: ShadowSubmission) bool {
                return a.key < b.key;
            }
        }.lessThan);
        self.shadows.render(self);
        self.shadow_params = packShadowParams(&self.shadows);
        self.forward.setShadows(self.shadows.atlasView(), self.shadows.compareSampler(), &self.shadow_params);
        if (self.mode == .deferred)
            // ShadowParams is generated per shader, but mesh's and deferred's are
            // byte-identical by construction (both come from the same block in
            // pbr_lib.glsl.inc), so the cast is safe. A shared type would make
            // that guarantee explicit rather than assumed.
            self.lit.setShadows(self.shadows.atlasView(), self.shadows.compareSampler(), @ptrCast(&self.shadow_params));

        // 2) Main geometry pass, replaying this frame's opaque submissions.
        switch (self.mode) {
            .deferred => {
                zupra.beginDrawingPass(self.gbuffer.pass());
                self.geo.begin(self.camera, self.gbuffer.passSignature());
            },
            .forward => {
                // Prepass first, into its own target: depth for early-z, normals
                // for AO. Both are sampled before the colour pass begins.
                if (self.depth_prepass) {
                    zupra.beginDrawingFramebufferClear(self.prepass_fb, .{ .r = 0, .g = 0, .b = 0, .a = 0 });
                    self.prepass.begin(self.camera, self.prepass_fb.passSignature());
                    for (self.opaque_queue.items) |e| {
                        if (!self.visible(e.bounds)) continue;
                        self.prepass.draw(e.mesh, e.model, e.material);
                    }
                    self.prepass.end();
                    zupra.endDrawing();

                    if (self.ssao_enabled) {
                        self.ssao.settings.temporal_jitter = self.isTaaActive();
                        self.ssao.render(
                            self.camera,
                            self.prepass_fb.depth_sample,
                            self.prepass_fb.sample_view,
                        );
                        self.forward.setSsao(self.ssao.aoView());
                    } else {
                        self.forward.setSsao(zupra.intern.white_1x1.view);
                    }
                } else {
                    // No prepass, no normals, no AO. White is the identity for
                    // the multiply in the shader.
                    self.forward.setSsao(zupra.intern.white_1x1.view);
                    zupra.beginDrawingFramebufferClear(self.scene_color, self.clear_color);
                    self.forward.beginEx(self.camera, self.env, self.scene_color.passSignature());
                }
                self.forward.depth_prepass = self.depth_prepass;

                // Colour pass. When the prepass ran, attach ITS depth and load
                // it; clearing would discard the buffer the pass exists to test
                // against. Without a prepass, fall back to the target's own.
                if (self.depth_prepass) {
                    zupra.beginDrawingFramebufferLoadDepth(
                        self.scene_color,
                        self.clear_color,
                        self.prepass_fb.depth_view,
                    );
                    self.forward.beginEx(
                        self.camera,
                        self.env,
                        self.scene_color.passSignatureWith(self.prepass_fb.depth_format),
                    );
                } else {
                    zupra.beginDrawingFramebufferClear(self.scene_color, self.clear_color);
                    self.forward.beginEx(self.camera, self.env, self.scene_color.passSignature());
                }
            },
        }
        for (self.opaque_queue.items) |e| {
            if (!self.visible(e.bounds)) continue;
            self.visible_draws += 1;
            self.drawOpaque(e.mesh, e.model, e.material);
        }

        // 3) Resolve lighting and composite.
        switch (self.mode) {
            .deferred => {
                self.geo.end();
                zupra.endDrawing(); // end G-buffer pass

                if (self.ssao_enabled) {
                    self.ssao.settings.temporal_jitter = self.isTaaActive();
                    self.ssao.render(self.camera, self.gbuffer.depthTexture().view, self.gbuffer.normalTexture().view);
                    self.lit.setSsao(self.ssao.aoView(), self.ssao.aoSampler());
                    self.forward.setSsao(self.ssao.aoView()); // FIXME: Should SSAO even be bound for transparents??
                } else {
                    self.lit.setSsao(zupra.intern.white_1x1.view, self.ssao.aoSampler());
                }

                // PBR opaque -> scene-color via the lighting pass.
                zupra.beginDrawingFramebufferClear(self.scene_color, self.clear_color);
                self.lit.render(self.gbuffer, self.camera, self.env, self.scene_color.passSignature());
                zupra.endDrawing(); // end lighting pass

                // Non-PBR opaque (unlit/lambert) -> scene-color, forward-shaded,
                // depth-tested AND written against the G-buffer depth so they merge
                // correctly with the PBR geometry.
                self.flushForwardOpaque();
            },
            .forward => {
                self.forward.end();
                zupra.endDrawing(); // end opaque scene-color pass
            },
        }

        self.composite();

        // Present: tonemap scene-color (HDR) onto the swapchain.
        const depth: ?zupra.graphics.texture.Texture = switch (self.mode) {
            .deferred => self.gbuffer.depthTexture(),
            .forward => if (self.depth_prepass) self.prepass_fb.depthTexture() else null,
        };
        var hdr = self.scene_color.asTexture();
        if (self.isTaaActive()) {
            self.velocity.begin(self.camera, self.opaqueDepthView());
            for (self.moved_queue.items) |e| {
                self.velocity.draw(e.mesh, e.model, e.prev_model, e.material);
            }
            self.velocity.end();
            hdr = self.taa.resolve(
                hdr,
                self.opaqueDepthSampleView(),
                if (self.velocity.any_drawn) self.velocity.velocityView() else null,
                self.camera.unjitteredViewProjection(),
            );
        }
        if (self.bloom_enabled) hdr = self.bloom.render(hdr);
        self.post.present(hdr, depth);
    }

    fn composite(self: *SceneRenderer) void {
        const has_sky = self.skybox != null;
        const has_transparent = self.transparent.items.len > 0;
        if (!has_sky and !has_transparent) return;

        const opaque_depth = switch (self.mode) {
            .deferred => self.gbuffer.depth_view,
            .forward => self.scene_color.depth_view,
        };

        var att = sg.Attachments{};
        att.colors[0] = self.scene_color.color_view;
        att.depth_stencil = opaque_depth;
        var action = sg.PassAction{};
        action.colors[0] = .{ .load_action = .LOAD };
        action.depth = .{ .load_action = .LOAD };

        zupra.beginDrawingPass(self.scene_color.passWith(.{
            .action = action,
            .depth_view = self.opaqueDepthView(),
            // .resolve defaults to true — final pass, so this is the one that
            // resolves MSAA into the image the post chain samples.
        }));
        const sig = self.scene_color.passSignatureWith(.DEPTH);

        if (self.envmap) |em| {
            em.render(self.camera, sig);
        } else if (self.skybox) |*s| {
            s.render(self.camera, sig);
        }

        // Then sorted transparents over everything.
        if (has_transparent) {
            std.mem.sort(TransparentEntry, self.transparent.items, {}, struct {
                fn farther(_: void, a: TransparentEntry, b: TransparentEntry) bool {
                    return a.depth > b.depth;
                }
            }.farther);
            self.forward.beginEx(self.camera, self.env, sig);
            for (self.transparent.items) |e| {
                if (!self.visible(e.bounds)) continue;
                self.visible_draws += 1;
                self.forward.draw(e.mesh, e.model, e.material);
            }
            self.forward.end();
        }

        zupra.endDrawing();
    }

    fn opaqueDepthView(self: SceneRenderer) sg.View {
        return switch (self.mode) {
            .deferred => self.gbuffer.depth_view,
            // Forward: with a prepass, opaque depth lives in the prepass target
            // and scene_color's own depth is never written -- the colour pass
            // attaches the prepass buffer rather than clearing its own. The sky
            // and transparents must test against the same buffer the opaques
            // actually filled.
            .forward => if (self.depth_prepass)
                self.prepass_fb.depth_view
            else
                self.scene_color.depth_view,
        };
    }

    /// The same depth buffer as opaqueDepthView, but as a texture view.
    fn opaqueDepthSampleView(self: SceneRenderer) sg.View {
        return switch (self.mode) {
            .deferred => self.gbuffer.depthTexture().view,
            .forward => if (self.depth_prepass)
                self.prepass_fb.depth_sample
            else
                self.scene_color.depth_sample,
        };
    }

    fn flushForwardOpaque(self: *SceneRenderer) void {
        if (self.forward_opaque.items.len == 0) return;

        var action = sg.PassAction{};
        action.colors[0] = .{ .load_action = .LOAD };
        action.depth = .{ .load_action = .LOAD };

        // Not the final pass into scene_color (composite still follows), so skip the
        // MSAA resolve here — composite's resolve captures everything.
        zupra.beginDrawingPass(self.scene_color.passWith(.{
            .action = action,
            .depth_view = self.opaqueDepthView(),
            .resolve = false,
        }));

        const sig = self.scene_color.passSignatureWith(.DEPTH);
        self.forward.beginEx(self.camera, self.env, sig);
        for (self.forward_opaque.items) |e| {
            self.forward.draw(e.mesh, e.model, e.material);
        }
        self.forward.end();
        zupra.endDrawing();
    }
};

fn packShadowParams(rend: *const ShadowRenderer) ShadowParams {
    var p: ShadowParams = std.mem.zeroes(ShadowParams);
    const data = rend.shadowData();
    const MV = shadow_max_views;
    for (data, 0..) |d, li| {
        var c: usize = 0;
        while (c < MV) : (c += 1) {
            const o = (li * MV + c) * 4;
            const m = d.view_proj[c]; // [16]f32, row order
            p.sh_vp[o + 0] = .{ m[0], m[1], m[2], m[3] };
            p.sh_vp[o + 1] = .{ m[4], m[5], m[6], m[7] };
            p.sh_vp[o + 2] = .{ m[8], m[9], m[10], m[11] };
            p.sh_vp[o + 3] = .{ m[12], m[13], m[14], m[15] };
            p.sh_rect[li * MV + c] = d.rect[c];
        }
        p.sh_info[li] = d.params;
        p.sh_split[li] = d.splits;
        p.sh_bias[li] = d.normal_bias;
        p.sh_pos[li] = d.pos_kind;
        p.sh_fade[li] = d.fade;
    }
    return p;
}
