//! src/render/deferred.zig
//!
//! Deferred rendering: GEOMETRY pass (fills the G-buffer) + LIGHTING pass
//! (fullscreen Cook-Torrance PBR over the G-buffer).
//!
//! Flow:
//!     // 1) geometry -> G-buffer
//!     zupra.beginDrawingPass(gbuf.pass());
//!     geo.begin(cam, gbuf.passSignature());
//!     geo.drawModel(inst);
//!     geo.end();
//!     zupra.endDrawing();
//!     // 2) lighting -> screen (or an HDR FBO)
//!     zupra.beginDrawingClear(black);
//!     lit.render(gbuf, cam, lights, ambient, PassSignature.swapchainPass());
//!     zupra.endDrawing();

const std = @import("std");
const sg = @import("sokol").gfx;

const gfx = @import("../graphics/graphics.zig");
const pipeline = @import("../graphics/pipeline.zig");
const math = @import("../math.zig");
const mesh_mod = @import("mesh.zig");
const model_mod = @import("model.zig");
const material_mod = @import("material.zig");
const gbuffer_mod = @import("gbuffer.zig");
const Camera3D = @import("camera3d.zig").Camera3D;
const zupra = @import("../root.zig");

const shd_geo = @import("shaders").gbuffer;
const shd_light = @import("shaders").deferred_lighting;

const Mesh = mesh_mod.Mesh;
const Material = material_mod.Material;
const ModelInstance = model_mod.ModelInstance;
const GBuffer = gbuffer_mod.GBuffer;
const ShaderProgram = gfx.ShaderProgram;
const Vertex2D = gfx.Vertex2D;
const PipelineCache = pipeline.PipelineCache;
const PipelineKey = pipeline.PipelineKey;
const PassSignature = pipeline.PassSignature;
const Matrix = math.Matrix;
const Vec3 = math.Vec3;
const Color = zupra.Color;
const Light = @import("light.zig").Light;
const LightParams = @import("light.zig").LightParams;
const packLightParams = @import("light.zig").packLightParams;
const MAX_LIGHTS = @import("light.zig").MAX_LIGHTS;
const Environment = @import("environment.zig").Environment;

// =====================================================================
// Geometry pass
// =====================================================================

const GeoVs = shd_geo.VsParams;
const GeoFs = shd_geo.FsParams;
const ReconParams = shd_light.ReconParams;

var geo_shader: ?ShaderProgram = null;

pub fn geoSharedShader() ShaderProgram {
    if (geo_shader == null) {
        geo_shader = ShaderProgram.init(shd_geo.gbufferShaderDesc, .{
            .layout = .mesh,
            .slots = .{ .vs_params = shd_geo.UB_vs_params, .fs_params = shd_geo.UB_fs_params },
        });
    }
    return geo_shader.?;
}

pub const GeometryRenderer = struct {
    cache: *PipelineCache,
    shader: ShaderProgram,
    pass: PassSignature = .{},
    material_sampler: sg.Sampler,
    view_proj: Matrix = undefined,
    active: bool = false,

    pub fn init(cache: *PipelineCache) GeometryRenderer {
        return .{
            .cache = cache,
            .shader = geoSharedShader(),
            .material_sampler = sg.makeSampler(.{
                .min_filter = .LINEAR,
                .mag_filter = .LINEAR,
                .mipmap_filter = .LINEAR,
                .wrap_u = .REPEAT,
                .wrap_v = .REPEAT,
            }),
        };
    }

    pub fn deinit(self: *GeometryRenderer) void {
        sg.destroySampler(self.material_sampler);
        self.active = false;
    }

    pub fn begin(self: *GeometryRenderer, camera: Camera3D, pass: PassSignature) void {
        std.debug.assert(!self.active);
        self.active = true;
        self.view_proj = camera.viewProjection();
        self.pass = pass;
    }

    pub fn drawMesh(self: *GeometryRenderer, mesh: Mesh, model: Matrix, material: Material) void {
        std.debug.assert(self.active);

        const key = PipelineKey{
            .shader = self.shader.handle,
            .layout = .mesh,
            .index_type = mesh.index_type,
            .pass = self.pass,
            .primitive = .TRIANGLES,
            .cull = material.cullMode(),
            .blend = .none,
            .depth_test = true,
            .depth_write = true,
            .face_winding = .CCW,
        };
        const pip = self.cache.get(key) catch |err| {
            std.log.err("GeometryRenderer: pipeline cache failed: {}", .{err});
            return;
        };

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = mesh.vbuf;
        bindings.index_buffer = mesh.ibuf;

        var vs = GeoVs{
            .model = @bitCast(model),
            .view_proj = @bitCast(self.view_proj),
            .uv_scale = .{ material.uv_scale[0], material.uv_scale[1], 0, 0 },
        };
        const c = material.base_color;
        var fs = GeoFs{
            .base_color = .{ c.r, c.g, c.b, c.a },
            .mat_params = .{ material.metallic, material.roughness, material.occlusion_strength, material.normal_scale },
            .emissive = .{ material.emissive.r, material.emissive.g, material.emissive.b, material.emissive_strength },
        };
        const mat_smp = material.sampler orelse self.material_sampler;

        bindings.views[shd_geo.VIEW_base_color_map] = material.map(.base_color).view;
        bindings.views[shd_geo.VIEW_normal_map] = material.map(.normal).view;
        bindings.views[shd_geo.VIEW_metallic_roughness_map] = material.map(.metallic_roughness).view;
        bindings.views[shd_geo.VIEW_occlusion_map] = material.map(.occlusion).view;
        bindings.views[shd_geo.VIEW_emissive_map] = material.map(.emissive).view;
        bindings.samplers[shd_geo.SMP_smp_material] = mat_smp;

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(self.shader.slots.vs_params, sg.asRange(&vs));
        sg.applyUniforms(self.shader.slots.fs_params.?, sg.asRange(&fs));

        var uvp = mesh_mod.uvParams(material);
        sg.applyUniforms(shd_geo.UB_uv_params, sg.asRange(&uvp));

        sg.draw(0, mesh.index_count, 1);
    }

    pub fn drawModel(self: *GeometryRenderer, inst: ModelInstance) void {
        const model_matrix = inst.modelMatrix();
        const model = inst.model;
        for (model.meshes, 0..) |submesh, i| {
            self.drawMesh(submesh, model_matrix, model.materials[model.mesh_material[i]]);
        }
    }

    pub fn end(self: *GeometryRenderer) void {
        std.debug.assert(self.active);
        self.active = false;
    }
};

pub const DeferredRenderer = struct {
    cache: *PipelineCache,
    shader: sg.Shader,
    sampler: sg.Sampler,
    fullscreen_vbuf: sg.Buffer,

    ibl_irradiance: sg.View = .{},
    ibl_prefilter: sg.View = .{},
    ibl_brdf_lut: sg.View = .{},
    ibl_sampler: sg.Sampler = .{},

    pub fn init(cache: *PipelineCache) DeferredRenderer {
        const shader = sg.makeShader(shd_light.deferredLightingShaderDesc(sg.queryBackend()));
        const sampler = sg.makeSampler(.{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        // Fullscreen triangle in clip space; uv chosen for this backend's
        // render-target origin so the sampled G-buffer isn't flipped.
        const top_left = sg.queryFeatures().origin_top_left;
        const clip = [3][2]f32{ .{ -1, -1 }, .{ 3, -1 }, .{ -1, 3 } };
        var verts: [3]Vertex2D = undefined;
        for (clip, 0..) |p, i| {
            const u = 0.5 + 0.5 * p[0];
            const v = if (top_left) 0.5 - 0.5 * p[1] else 0.5 + 0.5 * p[1];
            verts[i] = .{ .pos = p, .uv = .{ u, v }, .color = 0xFFFFFFFF };
        }
        const fullscreen_vbuf = sg.makeBuffer(.{
            .usage = .{ .vertex_buffer = true, .immutable = true },
            .data = sg.asRange(&verts),
        });

        return .{ .cache = cache, .shader = shader, .sampler = sampler, .fullscreen_vbuf = fullscreen_vbuf };
    }

    pub fn deinit(self: *DeferredRenderer) void {
        sg.destroyBuffer(self.fullscreen_vbuf);
        sg.destroySampler(self.sampler);
        sg.destroyShader(self.shader);
    }

    pub fn setIbl(self: *@This(), irradiance: sg.View, prefilter: sg.View, brdf_lut: sg.View, sampler: sg.Sampler) void {
        self.ibl_irradiance = irradiance;
        self.ibl_prefilter = prefilter;
        self.ibl_brdf_lut = brdf_lut;
        self.ibl_sampler = sampler;
    }

    pub fn render(
        self: *DeferredRenderer,
        gbuf: GBuffer,
        camera: Camera3D,
        env: Environment,
        pass: PassSignature,
    ) void {
        var params = packLightParams(env.lights(), env.ambient, camera.position);

        const inv_vp = zupra.math.zm.inverse(camera.viewProjection());
        var recon = ReconParams{ .inv_view_proj = @bitCast(inv_vp) };

        const key = PipelineKey{
            .shader = self.shader,
            .layout = .fullscreen, // fullscreen tri uses pos+uv (color slot unused)
            .index_type = .u32,
            .indexed = false,
            .pass = pass,
            .primitive = .TRIANGLES,
            .cull = .NONE,
            .blend = .none,
            .depth_test = false,
            .depth_write = false,
        };
        const pip = self.cache.get(key) catch |err| {
            std.log.err("DeferredRenderer: pipeline cache failed: {}", .{err});
            return;
        };

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = self.fullscreen_vbuf;
        bindings.views[shd_light.VIEW_tex_albedo] = gbuf.albedoTexture().view;
        bindings.views[shd_light.VIEW_tex_normal] = gbuf.normalTexture().view;
        bindings.views[shd_light.VIEW_tex_material] = gbuf.materialTexture().view;
        bindings.views[shd_light.VIEW_tex_emissive] = gbuf.emissiveTexture().view;
        bindings.views[shd_light.VIEW_tex_depth] = gbuf.depthTexture().view;
        bindings.samplers[shd_light.SMP_smp] = self.sampler;
        bindings.views[shd_light.VIEW_irradiance_map] = self.ibl_irradiance;
        bindings.views[shd_light.VIEW_prefilter_map] = self.ibl_prefilter;
        bindings.views[shd_light.VIEW_brdf_lut] = self.ibl_brdf_lut;
        bindings.samplers[shd_light.SMP_smp_cube] = self.ibl_sampler;

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(shd_light.UB_light_params, sg.asRange(&params));
        sg.applyUniforms(shd_light.UB_recon_params, sg.asRange(&recon));
        sg.draw(0, 3, 1);
    }
};

pub fn deinitShared() void {
    if (geo_shader) |*s| {
        s.deinit();
        geo_shader = null;
    }
}
