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

const VsParams = shd.VsParams;
const FsParams = extern struct {
    base_color: [4]f32,
    lights: LightParams,
};

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

// --- shared mesh shader ---

var shared_shader: ?ShaderProgram = null;

pub fn sharedShader() ShaderProgram {
    if (shared_shader == null) {
        shared_shader = ShaderProgram.init(shd.meshShaderDesc, .{
            .layout = .mesh,
            .slots = .{ .vs_params = shd.UB_vs_params, .fs_params = shd.UB_fs_params },
        });
    }
    return shared_shader.?;
}

pub fn deinitShared() void {
    if (shared_shader) |*s| {
        s.deinit();
        shared_shader = null;
    }
}

// --- draw path ---

pub const MeshRenderer = struct {
    cache: *PipelineCache,
    shader: ShaderProgram,
    pass: PassSignature = .{},

    // per-frame state captured at begin()
    view_proj: Matrix = undefined,
    lights: LightParams = undefined,
    active: bool = false,

    pub fn init(cache: *PipelineCache) MeshRenderer {
        return .{ .cache = cache, .shader = sharedShader() };
    }

    pub fn begin(self: *MeshRenderer, camera: Camera3D, env: Environment) void {
        self.beginEx(camera, env, PassSignature.swapchainPass());
    }

    pub fn beginEx(self: *MeshRenderer, camera: Camera3D, env: Environment, pass: PassSignature) void {
        std.debug.assert(!self.active);
        self.active = true;
        self.view_proj = camera.viewProjection();
        self.lights = packLightParams(env.lights, env.ambient, camera.position);
        self.pass = pass;
    }

    pub fn draw(self: *MeshRenderer, mesh: Mesh, model: Matrix, material: Material) void {
        std.debug.assert(self.active);

        const key = PipelineKey{
            .shader = self.shader.handle,
            .layout = .mesh,
            .index_type = mesh.index_type,
            .pass = self.pass,
            .primitive = .TRIANGLES,
            .cull = .NONE,
            .blend = .none,
            .depth_test = true,
            .depth_write = true,
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
        };
        var fs = FsParams{
            .base_color = .{ material.base_color.r, material.base_color.g, material.base_color.b, material.base_color.a },
            .lights = self.lights,
        };

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(self.shader.slots.vs_params, sg.asRange(&vs));
        sg.applyUniforms(self.shader.slots.fs_params.?, sg.asRange(&fs));
        sg.draw(0, mesh.index_count, 1);
    }

    pub fn end(self: *MeshRenderer) void {
        std.debug.assert(self.active);
        self.active = false;
    }
};
