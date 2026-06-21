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
const Material = material_mod.Material;
const ModelInstance = model_mod.ModelInstance;
const GBuffer = gbuffer_mod.GBuffer;
const GeometryRenderer = deferred_mod.GeometryRenderer;
const DeferredRenderer = deferred_mod.DeferredRenderer;
const Matrix = math.Matrix;
const Color = zupra.Color;
const Skybox = @import("skybox.zig").Skybox;
const Ibl = @import("ibl.zig").Ibl;

pub const ShadingMode = enum { forward, deferred };

const TransparentEntry = struct {
    mesh: Mesh,
    model: Matrix,
    material: Material,
    depth: f32, // squared distance to camera, for back-to-front sort
};

const ForwardEntry = struct {
    mesh: Mesh,
    model: Matrix,
    material: Material,
};

pub const SceneRenderer = struct {
    mode: ShadingMode,
    cache: *PipelineCache,
    allocator: std.mem.Allocator,
    width: u32,
    height: u32,
    clear_color: Color = .{ .r = 0.02, .g = 0.02, .b = 0.03, .a = 1 },

    scene_color: Framebuffer,
    present: Present,
    skybox: ?Skybox = null,
    ibl: ?Ibl = null,
    ibl_baked: bool = false,

    // deferred-only
    gbuffer: GBuffer = undefined,
    geo: GeometryRenderer = undefined,
    lit: DeferredRenderer = undefined,

    // forward-only
    forward: MeshRenderer = undefined,
    forward_opaque: std.ArrayListUnmanaged(ForwardEntry) = .empty,

    // transparent draw queue (both modes); flushed sorted in the NEXT step
    transparent: std.ArrayListUnmanaged(TransparentEntry) = .empty,

    // per-frame captured state
    camera: Camera3D = undefined,
    env: Environment = .{},

    pub fn init(allocator: std.mem.Allocator, cache: *PipelineCache, mode: ShadingMode, width: u32, height: u32) SceneRenderer {
        var self = SceneRenderer{
            .mode = mode,
            .cache = cache,
            .allocator = allocator,
            .width = width,
            .height = height,
            .scene_color = makeSceneColor(mode, width, height),
            .present = Present.init(cache),
        };
        // Forward renderer exists in BOTH modes: forward mode uses it for
        // opaque, and both modes use it to forward-shade transparents.
        self.forward = MeshRenderer.init(cache);
        if (mode == .deferred) {
            self.gbuffer = GBuffer.init(width, height);
            self.geo = GeometryRenderer.init(cache);
            self.lit = DeferredRenderer.init(cache);
        }
        self.skybox = Skybox.init(cache);
        self.ibl = Ibl.init(allocator, cache);
        return self;
    }

    pub fn deinit(self: *SceneRenderer) void {
        self.transparent.deinit(self.allocator);
        self.forward_opaque.deinit(self.allocator);
        if (self.mode == .deferred) {
            self.lit.deinit();
            self.gbuffer.deinit();
        }
        self.present.deinit();
        self.scene_color.deinit();
        if (self.skybox) |*s| s.deinit();
        if (self.ibl) |*ibl| ibl.deinit();
    }

    fn makeSceneColor(mode: ShadingMode, w: u32, h: u32) Framebuffer {
        return Framebuffer.init(.{
            .width = w,
            .height = h,
            .color_format = .RGBA16F,
            // forward writes opaque depth here; deferred lighting is fullscreen
            // (its depth lives in the G-buffer), so none needed yet.
            .depth_format = if (mode == .forward) .DEPTH else .NONE,
        });
    }

    fn ensureSize(self: *SceneRenderer, w: u32, h: u32) void {
        if (w == 0 or h == 0 or (w == self.width and h == self.height)) return;
        self.scene_color.deinit();
        self.scene_color = makeSceneColor(self.mode, w, h);
        if (self.mode == .deferred) {
            self.gbuffer.deinit();
            self.gbuffer = GBuffer.init(w, h);
        }
        self.width = w;
        self.height = h;
    }

    pub fn begin(self: *SceneRenderer, camera: Camera3D, env: Environment) void {
        const w: u32 = @intFromFloat(sapp.widthf());
        const h: u32 = @intFromFloat(sapp.heightf());
        self.ensureSize(w, h);

        self.camera = camera;
        self.camera.setViewport(@floatFromInt(w), @floatFromInt(h));
        self.env = env;
        self.transparent.clearRetainingCapacity();
        self.forward_opaque.clearRetainingCapacity();

        if (!self.ibl_baked) {
            if (self.ibl) |*ibl| {
                if (self.skybox) |*sky| {
                    ibl.bake(sky);
                    self.ibl_baked = true;
                }
            }
        }
        if (self.ibl) |ibl| {
            self.forward.setIbl(ibl.irradianceView(), ibl.prefilterView(), ibl.brdfLutView(), ibl.cubeSampler());
            if (self.mode == .deferred)
                self.lit.setIbl(ibl.irradianceView(), ibl.prefilterView(), ibl.brdfLutView(), ibl.cubeSampler());
        }

        switch (self.mode) {
            .deferred => {
                zupra.beginDrawingPass(self.gbuffer.pass());
                self.geo.begin(self.camera, self.gbuffer.passSignature());
            },
            .forward => {
                zupra.beginDrawingFramebufferClear(self.scene_color, self.clear_color);
                self.forward.beginEx(self.camera, self.env, self.scene_color.passSignature());
            },
        }
    }

    pub fn draw(self: *SceneRenderer, inst: ModelInstance) void {
        const model_matrix = inst.modelMatrix();
        const dx = self.camera.position.x - inst.position.x;
        const dy = self.camera.position.y - inst.position.y;
        const dz = self.camera.position.z - inst.position.z;
        const depth = dx * dx + dy * dy + dz * dz;

        const model = inst.model;
        for (model.meshes, 0..) |submesh, i| {
            const material = model.materials[model.mesh_material[i]];
            if (material.alpha_mode == .blend) {
                self.transparent.append(self.allocator, .{
                    .mesh = submesh,
                    .model = model_matrix,
                    .material = material,
                    .depth = depth,
                }) catch {};
            } else {
                self.drawOpaque(submesh, model_matrix, material);
            }
        }
    }

    fn drawOpaque(self: *SceneRenderer, mesh: Mesh, model: Matrix, material: Material) void {
        switch (self.mode) {
            .deferred => {
                if (material.shading == .pbr) {
                    self.geo.drawMesh(mesh, model, material); // G-buffer (PBR lighting)
                } else {
                    // Forward-shaded after the lighting pass (respects shading model).
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
        switch (self.mode) {
            .deferred => {
                self.geo.end();
                zupra.endDrawing(); // end G-buffer pass

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
        zupra.beginDrawingClear(self.clear_color);
        self.present.render(self.scene_color.asTexture(), PassSignature.swapchainPass());
        zupra.endDrawing();
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

        var sig = PassSignature{ .color_count = 1, .depth_format = .DEPTH, .sample_count = 1 };
        sig.color_formats[0] = self.scene_color.color_format;

        zupra.beginDrawingPass(.{ .action = action, .attachments = att });

        // Sky first (fills background; geometry occludes it via depth test).
        if (self.skybox) |*sky| sky.render(self.camera, sig);

        // Then sorted transparents over everything.
        if (has_transparent) {
            std.mem.sort(TransparentEntry, self.transparent.items, {}, struct {
                fn farther(_: void, a: TransparentEntry, b: TransparentEntry) bool {
                    return a.depth > b.depth;
                }
            }.farther);
            self.forward.beginEx(self.camera, self.env, sig);
            for (self.transparent.items) |e| self.forward.draw(e.mesh, e.model, e.material);
            self.forward.end();
        }

        zupra.endDrawing();
    }

    fn flushForwardOpaque(self: *SceneRenderer) void {
        if (self.forward_opaque.items.len == 0) return;

        var att = sg.Attachments{};
        att.colors[0] = self.scene_color.color_view;
        att.depth_stencil = self.gbuffer.depth_view;

        var action = sg.PassAction{};
        action.colors[0] = .{ .load_action = .LOAD };
        action.depth = .{ .load_action = .LOAD };

        var sig = PassSignature{ .color_count = 1, .depth_format = .DEPTH, .sample_count = 1 };
        sig.color_formats[0] = self.scene_color.color_format;

        zupra.beginDrawingPass(.{ .action = action, .attachments = att });
        self.forward.beginEx(self.camera, self.env, sig);
        for (self.forward_opaque.items) |e| {
            self.forward.draw(e.mesh, e.model, e.material);
        }
        self.forward.end();
        zupra.endDrawing();
    }
};
