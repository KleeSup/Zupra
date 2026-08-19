//! src/render/velocity.zig
//!
//! Per-object motion vector pass.
//!
//! TAA reprojects the history using camera motion, reconstructing each pixel's
//! world position from depth and pushing it through the previous
//! view-projection. That is exact for anything that did not move, which covers
//! most of a scene, but an object moving under its own transform has no
//! camera-derived velocity. Its history is fetched from the wrong place, and
//! what the neighbourhood clamp cannot hide appears as ghosting behind it.
//!
//! This pass records where each surface actually was, so the resolve can follow
//! it. The input it needs is the previous model matrix per object, which is why
//! this arrives with RenderWorld rather than before it. An immediate API
//! discards its list every frame and has nowhere to keep that matrix.
//!
//! Only objects that moved are drawn. A static object's velocity is entirely
//! accounted for by the camera reprojection TAA already does, so drawing it here
//! would write the same answer the resolve computes anyway. The target clears to
//! zero, and the resolve treats zero as "use camera reprojection".

const std = @import("std");
const sg = @import("sokol").gfx;
const zupra = @import("../root.zig");
const math = @import("../math.zig");
const pipeline = @import("../graphics/pipeline.zig");
const mesh_mod = @import("mesh.zig");
const fb = @import("framebuffer.zig");
const Camera3D = @import("camera3d.zig").Camera3D;

const shd = @import("shaders").velocity;

const Matrix = math.Matrix;
const Mesh = mesh_mod.Mesh;
const Material = @import("material.zig").Material;
const PipelineCache = pipeline.PipelineCache;
const PipelineKey = pipeline.PipelineKey;
const PassSignature = pipeline.PassSignature;
const Framebuffer = fb.Framebuffer;
const Texture = @import("../graphics/graphics.zig").texture.Texture;

const VelocityVsParams = extern struct {
    model: [16]f32,
    prev_model: [16]f32,
    view_proj: [16]f32,
    prev_view_proj: [16]f32,
};

const VelocityFsParams = extern struct {
    params: [4]f32, // origin_top_left, unused
};

pub const VelocityPass = struct {
    cache: *PipelineCache,
    shader: sg.Shader,
    target: Framebuffer = undefined,

    view_proj: Matrix = undefined,
    prev_view_proj: Matrix = undefined,
    sig: PassSignature = undefined,
    active: bool = false,
    /// Set once anything is drawn this frame. When nothing moved, the target is
    /// all zeros and the resolve can fall back to camera reprojection without
    /// sampling it.
    any_drawn: bool = false,

    width: u32 = 0,
    height: u32 = 0,

    pub fn init(cache: *PipelineCache, width: u32, height: u32) VelocityPass {
        var self = VelocityPass{
            .cache = cache,
            .shader = sg.makeShader(shd.velocityShaderDesc(sg.queryBackend())),
            .prev_view_proj = math.zm.identity(),
        };
        self.build(width, height);
        return self;
    }

    pub fn deinit(self: *VelocityPass) void {
        self.target.deinit();
        sg.destroyShader(self.shader);
    }

    pub fn resize(self: *VelocityPass, width: u32, height: u32) void {
        if (width == self.width and height == self.height) return;
        self.target.deinit();
        self.build(width, height);
    }

    fn build(self: *VelocityPass, width: u32, height: u32) void {
        self.target = Framebuffer.init(.{
            .width = width,
            .height = height,
            // Two signed channels of screen-space offset. RG16F rather than
            // RG8, because an 8 bit channel cannot represent a sub-pixel
            // movement, and sub-pixel movement is exactly what a temporal filter
            // is trying to resolve.
            .color_format = .RG16F,
            // Depth is loaded from the geometry pass rather than allocated here,
            // so only the nearest surface writes a velocity.
            .depth_format = .NONE,
        });
        self.width = width;
        self.height = height;
    }

    /// Begin recording. `depth_view` is the depth attachment the geometry pass
    /// filled, attached here so this pass writes only where the nearest surface
    /// is. Clearing to zero means anything not drawn reads as "did not move
    /// relative to the camera", which is the correct default.
    pub fn begin(self: *VelocityPass, camera: Camera3D, depth_view: sg.View) void {
        std.debug.assert(!self.active);
        self.active = true;
        self.any_drawn = false;
        self.view_proj = camera.viewProjection();

        zupra.beginDrawingFramebufferLoadDepth(
            self.target,
            .{ .r = 0, .g = 0, .b = 0, .a = 0 },
            depth_view,
        );
        self.sig = self.target.passSignatureWith(.DEPTH);
    }

    pub fn end(self: *VelocityPass) void {
        std.debug.assert(self.active);
        self.active = false;
        zupra.endDrawing();
        // Roll the camera forward here, after every draw this frame has used it.
        self.prev_view_proj = self.view_proj;
    }

    /// Record one submesh's motion. Only worth calling for objects whose
    /// transform actually changed.
    pub fn draw(self: *VelocityPass, mesh: Mesh, model: Matrix, prev_model: Matrix, material: Material) void {
        std.debug.assert(self.active);

        const key = PipelineKey{
            .shader = self.shader,
            .layout = .mesh,
            .index_type = mesh.index_type,
            .indexed = true,
            .pass = self.sig,
            .primitive = .TRIANGLES,
            .cull = material.cullMode(),
            .blend = .none,
            // Depth equal, no writes: the geometry pass already established the
            // visible surface, and this pass only annotates it. Testing LESS
            // would let a nearer surface that was culled from the main pass win
            // here and write a velocity for something not on screen.
            .depth_test = true,
            .depth_write = false,
            .face_winding = .CCW,
        };
        const pip = self.cache.get(key) catch |err| {
            std.log.err("VelocityPass: pipeline cache failed: {}", .{err});
            return;
        };

        var vs = VelocityVsParams{
            .model = @bitCast(model),
            .prev_model = @bitCast(prev_model),
            .view_proj = @bitCast(self.view_proj),
            .prev_view_proj = @bitCast(self.prev_view_proj),
        };
        var fs = VelocityFsParams{
            .params = .{ if (sg.queryFeatures().origin_top_left) 1.0 else 0.0, 0, 0, 0 },
        };

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = mesh.vbuf;
        bindings.index_buffer = mesh.ibuf;

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(shd.UB_vs_params, sg.asRange(&vs));
        sg.applyUniforms(shd.UB_velocity_params, sg.asRange(&fs));
        sg.draw(0, mesh.index_count, 1);

        self.any_drawn = true;
    }

    pub fn velocityView(self: VelocityPass) sg.View {
        return self.target.sample_view;
    }

    pub fn debugTexture(self: VelocityPass) Texture {
        return self.target.asTexture();
    }
};
