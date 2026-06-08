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

const VsParams = shd.VsParams;
const FsParams = shd.FsParams;

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

// --- material and light ---

pub const Material = struct {
    base_color: Color = .{ .r = 1, .g = 1, .b = 1, .a = 1 },
};

pub const DirectionalLight = struct {
    /// Direction the light travels (e.g. a sun pointing down is {0,-1,0}).
    /// The shader requires direction-to-light, so it's negated on upload.
    direction: Vec3 = .{ .x = -0.4, .y = -1, .z = -0.3 },
    color: Color = .{ .r = 1, .g = 1, .b = 1, .a = 1 },
    ambient: Color = .{ .r = 0.12, .g = 0.12, .b = 0.14, .a = 1 },
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
    pass: PassSignature = PassSignature.swapchain,

    // per-frame state captured at begin()
    view_proj: Matrix = undefined,
    light: DirectionalLight = .{},
    active: bool = false,

    pub fn init(cache: *PipelineCache) MeshRenderer {
        return .{ .cache = cache, .shader = sharedShader() };
    }

    pub fn begin(self: *MeshRenderer, camera: Camera3D, light: DirectionalLight) void {
        self.beginEx(camera, light, PassSignature.swapchain);
    }

    pub fn beginEx(self: *MeshRenderer, camera: Camera3D, light: DirectionalLight, pass: PassSignature) void {
        std.debug.assert(!self.active);
        self.active = true;
        self.view_proj = camera.viewProjection();
        self.light = light;
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
            .cull = .BACK,
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
            .base_color = colorVec4(material.base_color),
            .light_dir = dirVec4(self.light.direction.mul(-1)), // travel dir -> to-light dir
            .light_color = colorVec3(self.light.color),
            .ambient = colorVec3(self.light.ambient),
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

inline fn colorVec4(c: Color) [4]f32 {
    return .{ c.r, c.g, c.b, c.a };
}
inline fn colorVec3(c: Color) [4]f32 {
    return .{ c.r, c.g, c.b, 0 };
}
inline fn dirVec4(d: Vec3) [4]f32 {
    return .{ d.x, d.y, d.z, 0 };
}
