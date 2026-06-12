//! src/render/skybox.zig
//!
//! Procedural skybox. Renders a gradient sky + sun, either to the screen
//! (occluded by geometry via depth test) or into a cubemap face (for baking the
//! IBL environment). The appearance is data so it can be tweaked/animated.

const std = @import("std");
const sg = @import("sokol").gfx;
const math = @import("../math.zig");
const zm = math.zm;
const pipeline = @import("../graphics/pipeline.zig");
const FullscreenTriangle = @import("fullscreen.zig").FullscreenTriangle;
const Camera3D = @import("camera3d.zig").Camera3D;
const zupra = @import("../root.zig");

const shd = @import("shaders").sky;

const PipelineCache = pipeline.PipelineCache;
const PipelineKey = pipeline.PipelineKey;
const PassSignature = pipeline.PassSignature;
const Vec3 = math.Vec3;
const Matrix = math.Matrix;
const Color = zupra.Color;

const SkyParams = extern struct {
    inv_view_proj: [16]f32,
    camera_pos: [4]f32,
    sky_top: [4]f32,
    sky_horizon: [4]f32,
    sun_dir: [4]f32,
    sun_color: [4]f32,
};

pub const Skybox = struct {
    cache: *PipelineCache,
    shader: sg.Shader,
    tri: FullscreenTriangle,

    sky_top: Color = .{ .r = 0.18, .g = 0.34, .b = 0.65, .a = 1 },
    sky_horizon: Color = .{ .r = 0.65, .g = 0.72, .b = 0.82, .a = 1 },
    sun_direction: Vec3 = .{ .x = 0.4, .y = 0.8, .z = 0.45 },
    sun_sharpness: f32 = 512.0,
    sun_color: Color = .{ .r = 1.0, .g = 0.95, .b = 0.85, .a = 1 },
    sun_intensity: f32 = 8.0,

    pub fn init(cache: *PipelineCache) Skybox {
        return .{
            .cache = cache,
            .shader = sg.makeShader(shd.skyShaderDesc(sg.queryBackend())),
            .tri = FullscreenTriangle.init(),
        };
    }

    pub fn deinit(self: *Skybox) void {
        self.tri.deinit();
        sg.destroyShader(self.shader);
    }

    /// Draw to the screen: depth-tested against opaque depth (geometry occludes).
    pub fn render(self: *Skybox, camera: Camera3D, pass: PassSignature) void {
        self.renderRaw(zm.inverse(camera.viewProjection()), camera.position, pass, true);
    }

    /// Lower-level: render the sky for an explicit inverse view-projection.
    /// `depth_test` = true for the screen pass (occluded by geometry), false for
    /// baking into a cube face (no depth attachment, fill the whole face).
    pub fn renderRaw(self: *Skybox, inv_vp: Matrix, camera_pos: Vec3, pass: PassSignature, depth_test: bool) void {
        const key = PipelineKey{
            .shader = self.shader,
            .layout = .fullscreen,
            .index_type = .u32,
            .indexed = false,
            .pass = pass,
            .primitive = .TRIANGLES,
            .cull = .NONE,
            .blend = .none,
            .depth_test = depth_test,
            .depth_write = false,
        };
        const pip = self.cache.get(key) catch |err| {
            std.log.err("Skybox: pipeline cache failed: {}", .{err});
            return;
        };

        var params = SkyParams{
            .inv_view_proj = @bitCast(inv_vp),
            .camera_pos = .{ camera_pos.x, camera_pos.y, camera_pos.z, 0 },
            .sky_top = .{ self.sky_top.r, self.sky_top.g, self.sky_top.b, 0 },
            .sky_horizon = .{ self.sky_horizon.r, self.sky_horizon.g, self.sky_horizon.b, 0 },
            .sun_dir = .{ self.sun_direction.x, self.sun_direction.y, self.sun_direction.z, self.sun_sharpness },
            .sun_color = .{ self.sun_color.r, self.sun_color.g, self.sun_color.b, self.sun_intensity },
        };

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = self.tri.vbuf;

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(shd.UB_sky_params, sg.asRange(&params));
        sg.draw(0, 3, 1);
    }
};
