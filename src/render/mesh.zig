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
const Material = @import("material.zig").Material;
const Light = @import("light.zig").Light;
const LightParams = @import("light.zig").LightParams;
const packLightParams = @import("light.zig").packLightParams;
const Environment = @import("environment.zig").Environment;
const ShadingModel = @import("material.zig").ShadingModel;

const VsParams = shd.VsParams;
const PbrFs = extern struct {
    base_color: [4]f32,
    materials: [4]f32, // metallic, roughness, ao, unused
    lights: LightParams,
};
const LambertFs = extern struct {
    base_color: [4]f32,
    lights: LightParams, // camera_pos + ambient_count + light arrays (lambert ignores camera_pos)
};
const UnlitFs = extern struct {
    base_color: [4]f32,
};

const ShaderSet = struct {
    pbr: ShaderProgram,
    lambert: ShaderProgram,
    unlit: ShaderProgram,
};

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

    view_proj: Matrix = undefined,
    lights: LightParams = undefined,
    active: bool = false,

    ibl_irradiance: sg.View = .{},
    ibl_prefilter: sg.View = .{},
    ibl_brdf_lut: sg.View = .{},
    ibl_sampler: sg.Sampler = .{},

    pub fn init(cache: *PipelineCache) MeshRenderer {
        return .{ .cache = cache, .shaders = sharedShaders() };
    }

    pub fn setIbl(self: *MeshRenderer, irradiance: sg.View, prefilter: sg.View, brdf_lut: sg.View, sampler: sg.Sampler) void {
        self.ibl_irradiance = irradiance;
        self.ibl_prefilter = prefilter;
        self.ibl_brdf_lut = brdf_lut;
        self.ibl_sampler = sampler;
    }

    pub fn begin(self: *MeshRenderer, camera: Camera3D, env: Environment) void {
        self.beginEx(camera, env, PassSignature.swapchainPass());
    }

    pub fn beginEx(self: *MeshRenderer, camera: Camera3D, env: Environment, pass: PassSignature) void {
        std.debug.assert(!self.active);
        self.active = true;
        self.view_proj = camera.viewProjection();
        self.lights = packLightParams(env.lights(), env.ambient, camera.position);
        self.pass = pass;
    }

    pub fn draw(self: *MeshRenderer, mesh: Mesh, model: Matrix, material: Material) void {
        std.debug.assert(self.active);

        // Pick the shader for this material's shading model.
        const shader: ShaderProgram = switch (material.shading) {
            .pbr => self.shaders.pbr,
            .lambert => self.shaders.lambert,
            .unlit => self.shaders.unlit,
        };

        const key = PipelineKey{
            .shader = shader.handle,
            .layout = .mesh,
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

        // IBL bindings only exist on the PBR shader.
        if (material.shading == .pbr) {
            bindings.views[shd.VIEW_irradiance_map] = self.ibl_irradiance;
            bindings.views[shd.VIEW_prefilter_map] = self.ibl_prefilter;
            bindings.views[shd.VIEW_brdf_lut] = self.ibl_brdf_lut;
            bindings.samplers[shd.SMP_smp_cube] = self.ibl_sampler;
        }

        var vs = VsParams{
            .model = @bitCast(model),
            .view_proj = @bitCast(self.view_proj),
        };
        const c = material.base_color;

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(shader.slots.vs_params, sg.asRange(&vs));

        // Upload the fs uniforms matching the chosen shader.
        switch (material.shading) {
            .pbr => {
                var fs = PbrFs{
                    .base_color = .{ c.r, c.g, c.b, c.a },
                    .materials = .{ material.metallic, material.roughness, material.occlusion_strength, 0 },
                    .lights = self.lights,
                };
                sg.applyUniforms(shader.slots.fs_params.?, sg.asRange(&fs));
            },
            .lambert => {
                var fs = LambertFs{
                    .base_color = .{ c.r, c.g, c.b, c.a },
                    .lights = self.lights,
                };
                sg.applyUniforms(shader.slots.fs_params.?, sg.asRange(&fs));
            },
            .unlit => {
                var fs = UnlitFs{ .base_color = .{ c.r, c.g, c.b, c.a } };
                sg.applyUniforms(shader.slots.fs_params.?, sg.asRange(&fs));
            },
        }

        sg.draw(0, mesh.index_count, 1);
    }

    pub fn end(self: *MeshRenderer) void {
        std.debug.assert(self.active);
        self.active = false;
    }
};
