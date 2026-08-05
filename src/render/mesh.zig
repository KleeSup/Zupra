//! src/render/mesh.zig
//!
//! 3D mesh resource + a simple forward draw path.
//!
//!   * Mesh: immutable vertex buffer (Vertex3D) + index buffer. The index
//!     buffer is built from an IndexData union (u16 or u32) and then uploaded once,
//!     after which the mesh stores only the resolved IndexType tag. That tag
//!     flows into the PipelineKey, so a u16 mesh and a u32 mesh transparently
//!     get different pipelines from the cache. This is the u16/u32 switch doing
//!     real work for the first time.
//!
//!   * MeshRenderer: not a batcher, because meshes draw individually (3D gets
//!     instanced later, not sprite-batched). begin() captures the camera's
//!     view-projection and the light once; draw() renders one mesh with its
//!     own model matrix + material.

const std = @import("std");
const sg = @import("sokol").gfx;
const zupra = @import("../root.zig");

const gfx = @import("../graphics/graphics.zig");
const pipeline = @import("../graphics/pipeline.zig");
const math = @import("../math.zig");
const material_mod = @import("material.zig");
const lighting_mod = @import("lighting.zig");
const Camera3D = @import("camera3d.zig").Camera3D;

const shd = @import("shaders").mesh;
const shd_unlit = @import("shaders").unlit;
const shd_lambert = @import("shaders").lambert;

const Vertex3D = gfx.Vertex3D;
const IndexData = gfx.IndexData;
const IndexType = gfx.IndexType;
const ShaderProgram = gfx.ShaderProgram;
const PipelineCache = pipeline.PipelineCache;
const PipelineKey = pipeline.PipelineKey;
const PassSignature = pipeline.PassSignature;
const Matrix = math.Matrix;
const Vec3 = math.Vec3;
const Color = zupra.Color;
const Material = material_mod.Material;
const Light = @import("light.zig").Light;
const Environment = @import("environment.zig").Environment;
const ShadingModel = material_mod.ShadingModel;

const VsParams = shd.VsParams;
const UvParams = shd.UvParams;
const PbrFs = shd.FsParams;
const LambertFs = shd_lambert.FsParams;
const UnlitFs = shd_unlit.FsParams;

const ShaderSet = struct {
    pbr: ShaderProgram,
    lambert: ShaderProgram,
    unlit: ShaderProgram,
};

pub fn uvParams(m: Material) UvParams {
    var p: UvParams = undefined;
    inline for (0..material_mod.map_slot_count) |i| {
        const x = m.uv_xforms[i];
        const bit: u8 = @as(u8, 1) << @intCast(i);
        const set: f32 = if ((m.uv_set & bit) != 0) 1.0 else 0.0;
        p.uv_m[i] = x.m;
        p.uv_aux[i] = .{ x.offset[0], x.offset[1], set, 0 };
    }
    return p;
}

var shared_set: ?ShaderSet = null;

pub fn sharedShaders() ShaderSet {
    if (shared_set == null) {
        shared_set = .{
            .pbr = ShaderProgram.init(shd.meshShaderDesc, .{
                .layout = .mesh,
                .slots = .{ .vs_params = shd.UB_vs_params, .fs_params = shd.UB_fs_params },
            }),
            .lambert = ShaderProgram.init(shd_lambert.lambertShaderDesc, .{
                .layout = .mesh,
                .slots = .{ .vs_params = shd_lambert.UB_vs_params, .fs_params = shd_lambert.UB_fs_params },
            }),
            .unlit = ShaderProgram.init(shd_unlit.unlitShaderDesc, .{
                .layout = .mesh,
                .slots = .{ .vs_params = shd_unlit.UB_vs_params, .fs_params = shd_unlit.UB_fs_params },
            }),
        };
    }
    return shared_set.?;
}

pub fn deinitShared() void {
    if (shared_set) |*s| {
        s.pbr.deinit();
        s.lambert.deinit();
        s.unlit.deinit();
        shared_set = null;
    }
}

// --- mesh resource ---

pub const Mesh = struct {
    vbuf: sg.Buffer,
    ibuf: sg.Buffer,
    index_count: u32,
    index_type: IndexType,

    /// `vertices` and `indices` are copied into immutable GPU buffers, so the
    /// caller's data may be freed afterward. The index width is taken from the
    /// active union arm.
    pub fn init(vertices: []const Vertex3D, indices: IndexData) Mesh {
        const vbuf = sg.makeBuffer(.{
            .usage = .{ .vertex_buffer = true, .immutable = true },
            .data = sg.asRange(vertices),
        });

        const idx_bytes: []const u8 = switch (indices) {
            inline else => |s| std.mem.sliceAsBytes(s),
        };
        const idx_count: u32 = switch (indices) {
            inline else => |s| @intCast(s.len),
        };

        const ibuf = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true, .immutable = true },
            .data = .{ .ptr = idx_bytes.ptr, .size = idx_bytes.len },
        });

        return .{
            .vbuf = vbuf,
            .ibuf = ibuf,
            .index_count = idx_count,
            .index_type = std.meta.activeTag(indices),
        };
    }

    pub fn deinit(self: Mesh) void {
        sg.destroyBuffer(self.ibuf);
        sg.destroyBuffer(self.vbuf);
    }
};

// --- draw path ---

pub const MeshRenderer = struct {
    cache: *PipelineCache,
    shaders: ShaderSet,
    pass: PassSignature = .{},
    material_sampler: sg.Sampler,

    view_proj: Matrix = undefined,
    lighting: *lighting_mod.LightingFrame = undefined,
    active: bool = false,

    ibl_irradiance: sg.View = .{},
    ibl_prefilter: sg.View = .{},
    ibl_brdf_lut: sg.View = .{},
    ibl_sampler: sg.Sampler = .{},

    shadow_atlas: sg.View = .{},
    shadow_sampler: sg.Sampler = .{},
    shadow_params: *const shd.ShadowParams = undefined,

    pub fn init(cache: *PipelineCache) MeshRenderer {
        return .{
            .cache = cache,
            .shaders = sharedShaders(),
            .material_sampler = sg.makeSampler(.{
                .min_filter = .LINEAR,
                .mag_filter = .LINEAR,
                .mipmap_filter = .LINEAR,
                .wrap_u = .REPEAT,
                .wrap_v = .REPEAT,
            }),
        };
    }

    pub fn deinit(self: *MeshRenderer) void {
        sg.destroySampler(self.material_sampler);
    }

    pub fn setIbl(self: *MeshRenderer, irradiance: sg.View, prefilter: sg.View, brdf_lut: sg.View, sampler: sg.Sampler) void {
        self.ibl_irradiance = irradiance;
        self.ibl_prefilter = prefilter;
        self.ibl_brdf_lut = brdf_lut;
        self.ibl_sampler = sampler;
    }

    pub fn setShadows(self: *MeshRenderer, atlas: sg.View, sampler: sg.Sampler, params: *const shd.ShadowParams) void {
        self.shadow_atlas = atlas;
        self.shadow_sampler = sampler;
        self.shadow_params = params;
    }

    pub fn begin(self: *MeshRenderer, camera: Camera3D, env: *Environment) void {
        self.beginEx(camera, env, PassSignature.swapchainPass());
    }

    pub fn beginEx(self: *MeshRenderer, camera: Camera3D, env: *Environment, pass: PassSignature) void {
        std.debug.assert(!self.active);
        self.active = true;
        self.view_proj = camera.viewProjection();
        self.lighting = &env.lighting;
        self.pass = pass;
    }

    pub fn draw(self: *MeshRenderer, mesh: Mesh, model: Matrix, material: Material) void {
        std.debug.assert(self.active);

        // Pick the shader for this material's shading model.
        const shader: ShaderProgram = material.shader orelse switch (material.shading) {
            .pbr => self.shaders.pbr,
            .lambert => self.shaders.lambert,
            .unlit => self.shaders.unlit,
        };

        const key = PipelineKey{
            .shader = shader.handle,
            .layout = if (material.shader) |s| s.layout else if (material.shading == .unlit) .mesh_unlit else .mesh,

            .index_type = mesh.index_type,
            .pass = self.pass,
            .primitive = .TRIANGLES,
            .cull = material.cullMode(),
            .blend = material.blendMode(),
            .depth_test = true,
            .depth_write = material.alpha_mode != .blend,
            .face_winding = .CCW,
        };
        const pip = self.cache.get(key) catch |err| {
            std.log.err("MeshRenderer: pipeline cache failed: {}", .{err});
            return;
        };

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = mesh.vbuf;
        bindings.index_buffer = mesh.ibuf;

        var vs = VsParams{
            .model = @bitCast(model),
            .view_proj = @bitCast(self.view_proj),
            .uv_scale = .{ material.uv_scale[0], material.uv_scale[1], 0, 0 },
        };

        const bc = material.base_color;
        const emc = material.emissive;
        const es = material.emissive_strength;

        // Per-model: set every texture/sampler binding the shader declares, and
        // build the matching fs uniform struct. Nothing is applied yet.
        var pbr_fs: PbrFs = undefined;
        var lambert_fs: LambertFs = undefined;
        var unlit_fs: UnlitFs = undefined;
        const mat_smp = material.sampler orelse self.material_sampler;

        if (material.shader != null) {
            // Custom shader: the contract is MapSlot order, so bind by enum index.
            // material_surface.glsl.inc declares these at matching binding= slots.
            inline for (std.meta.fields(material_mod.MapSlot)) |f| {
                const slot: material_mod.MapSlot = @enumFromInt(f.value);
                bindings.views[f.value] = material.map(slot).view;
            }
            bindings.samplers[0] = mat_smp;
        } else {
            switch (material.shading) {
                .unlit => {
                    bindings.views[shd_unlit.VIEW_base_color_map] = material.map(.base_color).view;
                    bindings.views[shd_unlit.VIEW_emissive_map] = material.map(.emissive).view;
                    bindings.samplers[shd_unlit.SMP_smp_material] = mat_smp;

                    unlit_fs = .{
                        .base_color = .{ bc.r, bc.g, bc.b, bc.a },
                        .emissive = .{ emc.r, emc.g, emc.b, es },
                    };
                },
                .lambert => {
                    bindings.views[shd_lambert.VIEW_normal_map] = material.map(.normal).view;
                    bindings.views[shd_lambert.VIEW_base_color_map] = material.map(.base_color).view;
                    bindings.views[shd_lambert.VIEW_emissive_map] = material.map(.emissive).view;
                    bindings.samplers[shd_lambert.SMP_smp_material] = mat_smp;

                    bindings.views[shd_lambert.VIEW_shadow_atlas] = self.shadow_atlas;
                    bindings.samplers[shd_lambert.SMP_smp_shadow] = self.shadow_sampler;

                    self.lighting.bind(&bindings, .{
                        .light_data = shd_lambert.VIEW_light_data,
                        .cluster_table = shd_lambert.VIEW_cluster_table,
                        .cluster_indices = shd_lambert.VIEW_cluster_indices,
                        .sampler = shd_lambert.SMP_smp_data,
                    });

                    lambert_fs = .{
                        .base_color = .{ bc.r, bc.g, bc.b, bc.a },
                        .material = .{ 0, 0, 0, if (material.normal_flip_y) -material.normal_scale else material.normal_scale },
                        .emissive = .{ emc.r, emc.g, emc.b, es },
                    };
                },
                .pbr => {
                    bindings.views[shd.VIEW_irradiance_map] = self.ibl_irradiance;
                    bindings.views[shd.VIEW_prefilter_map] = self.ibl_prefilter;
                    bindings.views[shd.VIEW_brdf_lut] = self.ibl_brdf_lut;
                    bindings.samplers[shd.SMP_smp_cube] = self.ibl_sampler;
                    bindings.views[shd.VIEW_normal_map] = material.map(.normal).view;
                    bindings.views[shd.VIEW_base_color_map] = material.map(.base_color).view;
                    bindings.views[shd.VIEW_emissive_map] = material.map(.emissive).view;
                    bindings.views[shd.VIEW_metallic_roughness_map] = material.map(.metallic_roughness).view;
                    bindings.views[shd.VIEW_occlusion_map] = material.map(.occlusion).view;
                    bindings.samplers[shd.SMP_smp_material] = mat_smp;

                    bindings.views[shd.VIEW_shadow_atlas] = self.shadow_atlas;
                    bindings.samplers[shd.SMP_smp_shadow] = self.shadow_sampler;

                    self.lighting.bind(&bindings, .{
                        .light_data = shd.VIEW_light_data,
                        .cluster_table = shd.VIEW_cluster_table,
                        .cluster_indices = shd.VIEW_cluster_indices,
                        .sampler = shd.SMP_smp_data,
                    });

                    pbr_fs = .{
                        .base_color = .{ bc.r, bc.g, bc.b, bc.a },
                        .material = .{ material.metallic, material.roughness, material.occlusion_strength, if (material.normal_flip_y) -material.normal_scale else material.normal_scale },
                        .emissive = .{ emc.r, emc.g, emc.b, es },
                    };
                },
            }
        }

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);

        if (material.shader) |prog| {
            // vs_params is universal -> every mesh shader needs the transforms.
            sg.applyUniforms(prog.slots.vs_params, sg.asRange(&vs));

            if (prog.slots.uv_params) |slot| {
                var uvp = uvParams(material);
                sg.applyUniforms(slot, sg.asRange(&uvp));
            }
            if (prog.slots.fs_params) |slot| {
                if (material.shader_uniform_len > 0) {
                    sg.applyUniforms(slot, sg.asRange(material.shader_uniforms[0..material.shader_uniform_len]));
                } else {
                    std.log.err("Material custom shader has no uniforms set. Be sure to call setShaderUniforms()!", .{});
                    return; // skip the draw rather than let sokol panic
                }
            }
        } else {
            sg.applyUniforms(shader.slots.vs_params, sg.asRange(&vs));
            switch (material.shading) {
                .unlit => {
                    sg.applyUniforms(shader.slots.fs_params.?, sg.asRange(&unlit_fs));
                    var uvp = uvParams(material);
                    sg.applyUniforms(shd_unlit.UB_uv_params, sg.asRange(&uvp));
                },
                .lambert => {
                    sg.applyUniforms(shader.slots.fs_params.?, sg.asRange(&lambert_fs));
                    var uvp = uvParams(material);
                    sg.applyUniforms(shd_lambert.UB_uv_params, sg.asRange(&uvp));
                    var lp = self.lighting.params();
                    sg.applyUniforms(shd_lambert.UB_light_params, sg.asRange(&lp));
                    sg.applyUniforms(shd_lambert.UB_shadow_params, sg.asRange(self.shadow_params));
                },
                .pbr => {
                    sg.applyUniforms(shader.slots.fs_params.?, sg.asRange(&pbr_fs));
                    var uvp = uvParams(material);
                    sg.applyUniforms(shd.UB_uv_params, sg.asRange(&uvp));
                    var lp = self.lighting.params();
                    sg.applyUniforms(shd.UB_light_params, sg.asRange(&lp));
                    sg.applyUniforms(shd.UB_shadow_params, sg.asRange(self.shadow_params));
                },
            }
        }

        sg.draw(0, mesh.index_count, 1);
    }

    pub fn end(self: *MeshRenderer) void {
        std.debug.assert(self.active);
        self.active = false;
    }
};
