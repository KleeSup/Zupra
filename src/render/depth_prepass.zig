//! src/render/depth_prepass.zig
//!
//! Depth-only prepass for the forward path.
//!
//! Clustered shading keeps per-pixel cost tied to the lights in a froxel rather
//! than the lights in the scene, but it does nothing about OVERDRAW. Without a
//! prepass every fragment of every triangle runs the full shading path — cluster
//! table fetch, per-light Cook-Torrance, IBL cubemap samples, shadow PCF taps —
//! and then most of that work is discarded the moment a nearer surface writes
//! over it. In a scene of large overlapping objects the same pixel can be shaded
//! five or ten times, and the waste scales with light count, so it looks exactly
//! like a light-scaling problem when it isn't one.
//!
//! Laying depth down first with a position-only shader lets the shading pass run
//! with depth writes off and a LESS_EQUAL test, so only fragments that survive
//! ever reach the expensive shader. Clustering plus this prepass is what
//! "Forward+" normally refers to. The deferred path needs none of it — its
//! G-buffer pass already resolves visibility before any lighting runs.
//!
//! The prepass runs INSIDE the main colour pass with the colour write mask off,
//! not as a separate pass. That keeps the depth buffer live between the two
//! without a load action or a second attachment, and saves a pass switch.
//!
//! NOT EVERY DRAW BELONGS HERE. See mesh.depthPrepassEligible: blended geometry
//! has no single depth to write, alpha-masked geometry needs a fragment shader
//! to discard, and unlit or custom-shader materials may not use the .mesh vertex
//! layout this pass binds. Anything excluded keeps writing its own depth in the
//! shading pass exactly as before, so correctness never depends on the prepass
//! being enabled.

const std = @import("std");
const sg = @import("sokol").gfx;
const math = @import("../math.zig");
const pipeline = @import("../graphics/pipeline.zig");
const mesh_mod = @import("mesh.zig");
const Camera3D = @import("camera3d.zig").Camera3D;

const shd_prepass = @import("shaders").depth_prepass;
const shd_prepass_skinned = @import("shaders").depth_prepass_skinned;
const skeletal = @import("skeletal.zig");

const Matrix = math.Matrix;
const Mesh = mesh_mod.Mesh;
const Material = @import("material.zig").Material;
const PipelineCache = pipeline.PipelineCache;
const PipelineKey = pipeline.PipelineKey;
const PassSignature = pipeline.PassSignature;

pub const DepthPrepass = struct {
    cache: *PipelineCache,
    /// depth_prepass.glsl, NOT shadow_depth.glsl. The two look interchangeable
    /// -- both are position-in, depth-out -- but shadow_depth takes a single
    /// premultiplied mvp, and this pass has to reproduce mesh.glsl's two-step
    /// transform bit for bit or the shading pass z-fights against its own
    /// depth. See the header comment in depth_prepass.glsl.
    shader: sg.Shader,
    skinned_shader: sg.Shader,
    material_sampler: sg.Sampler,

    view_proj: Matrix = undefined,
    pass: PassSignature = undefined,
    active: bool = false,

    pub fn init(cache: *PipelineCache) DepthPrepass {
        const shader = sg.makeShader(shd_prepass.depthPrepassShaderDesc(sg.queryBackend()));
        const skinned_shader = sg.makeShader(shd_prepass_skinned.depthPrepassSkinnedShaderDesc(sg.queryBackend()));
        const state = sg.queryShaderState(shader);
        if (state != .VALID) {
            std.log.err(
                "DepthPrepass: depth_prepass shader is {s} (not VALID) — no depth will be laid down",
                .{@tagName(state)},
            );
        }
        return .{
            .cache = cache,
            .shader = shader,
            .skinned_shader = skinned_shader,
            .material_sampler = sg.makeSampler(.{
                .min_filter = .LINEAR,
                .mag_filter = .LINEAR,
                .mipmap_filter = .LINEAR,
                .wrap_u = .REPEAT,
                .wrap_v = .REPEAT,
            }),
        };
    }

    pub fn deinit(self: *DepthPrepass) void {
        sg.destroySampler(self.material_sampler);
        sg.destroyShader(self.skinned_shader);
        sg.destroyShader(self.shader);
    }

    /// Begin recording. `pass` must be the SAME signature the shading pass uses:
    /// the prepass draws inside that pass, so its pipelines have to agree with
    /// the attachment formats and sample count or pipeline creation fails.
    pub fn begin(self: *DepthPrepass, camera: Camera3D, pass: PassSignature) void {
        std.debug.assert(!self.active);
        self.active = true;
        self.view_proj = camera.viewProjection();
        self.pass = pass;
    }

    pub fn end(self: *DepthPrepass) void {
        std.debug.assert(self.active);
        self.active = false;
    }

    /// Lay down depth for one submesh. Silently skips anything the shading pass
    /// will write depth for itself.
    pub fn draw(self: *DepthPrepass, mesh: Mesh, model: Matrix, material: Material) void {
        self.drawInternal(mesh, model, material, null);
    }

    pub fn drawSkinned(self: *DepthPrepass, mesh: Mesh, model: Matrix, material: Material, skin: skeletal.Binding) void {
        self.drawInternal(mesh, model, material, skin);
    }

    fn drawInternal(self: *DepthPrepass, mesh: Mesh, model: Matrix, material: Material, skin: ?skeletal.Binding) void {
        std.debug.assert(self.active);
        if (!mesh_mod.depthPrepassEligible(material)) return;
        if (mesh.isSkinned() and skin == null) return;
        if (!mesh.isSkinned() and skin != null) return;

        // Cull mode and winding must match the shading pass draw for this same
        // mesh. If they diverge the two passes rasterise different triangles,
        // and the depth recorded here stops corresponding to what gets shaded —
        // surfaces drop out or z-fight depending on which way the mismatch runs.
        const key = PipelineKey{
            .shader = if (skin != null) self.skinned_shader else self.shader,
            .layout = if (skin != null) .mesh_skinned else .mesh,
            .index_type = mesh.index_type,
            .indexed = true,
            .pass = self.pass,
            .primitive = .TRIANGLES,
            .cull = material.cullMode(),
            .blend = .none,
            // Colour writes ON: this pass owns its own target now and emits the
            // world normal into it. It used to run inside the shading pass with
            // writes masked off, which worked only while it produced nothing.
            .color_write_mask = .RGBA,
            .depth_test = true,
            .depth_write = true,
            .face_winding = .CCW,
        };
        const pip = self.cache.get(key) catch |err| {
            std.log.err("DepthPrepass: pipeline cache failed: {}", .{err});
            return;
        };

        // Two separate matrices multiplied in the shader, exactly as mesh.glsl
        // does. Combining them here would be the same value in exact arithmetic
        // and a different one in floats, which shows up as speckled holes
        // wherever the prepass rounds nearer than the shading pass.
        var vs = shd_prepass.VsParams{
            .model = @bitCast(model),
            .view_proj = @bitCast(self.view_proj),
            .uv_scale = .{ material.uv_scale[0], material.uv_scale[1], 0, 0 },
        };
        const base_color = material.base_color;
        var fs = shd_prepass.FsParams{
            .base_color = .{ base_color.r, base_color.g, base_color.b, base_color.a },
            .alpha_params = material.alphaTestParams(),
        };

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = mesh.vbuf;
        bindings.index_buffer = mesh.ibuf;
        if (skin) |binding| {
            bindings.views[skeletal.palette_view_slot] = binding.palette;
            bindings.samplers[skeletal.palette_sampler_slot] = binding.sampler;
        }
        bindings.views[shd_prepass.VIEW_base_color_map] = material.map(.base_color).view;
        bindings.samplers[shd_prepass.SMP_smp_material] = material.sampler orelse self.material_sampler;

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(shd_prepass.UB_vs_params, sg.asRange(&vs));
        sg.applyUniforms(shd_prepass.UB_fs_params, sg.asRange(&fs));
        var uvp = mesh_mod.uvParams(material);
        sg.applyUniforms(shd_prepass.UB_uv_params, sg.asRange(&uvp));
        sg.draw(0, mesh.index_count, 1);
    }
};
