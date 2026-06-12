//! src/render/skybox.zig
//!
//! Procedural skybox. Renders a gradient sky + sun as the background, into
//! scene-color, depth-tested against the opaque depth so geometry occludes it.
//! Later it will also serve as the source environment baked into the IBL maps.
//!
//! Appearance is data on the struct (sky colors, sun) so it can be tweaked or
//! animated (day/night) freely.

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
const Color = zupra.Color;

const SkyParams = extern struct {
    inv_view_proj: [16]f32,
    camera_pos: [4]f32,
    sky_top: [4]f32,
    sky_horizon: [4]f32,
    sun_dir: [4]f32, // xyz dir-to-sun, w = sharpness
    sun_color: [4]f32, // rgb, w = intensity
};

pub const Skybox = struct {
    cache: *PipelineCache,
    shader: sg.Shader,
    tri: FullscreenTriangle,

    // appearance (linear HDR — this renders into the linear scene-color target)
    sky_top: Color = .{ .r = 0.18, .g = 0.34, .b = 0.65, .a = 1 },
    sky_horizon: Color = .{ .r = 0.65, .g = 0.72, .b = 0.82, .a = 1 },
    sun_direction: Vec3 = .{ .x = 0.4, .y = 0.8, .z = 0.45 }, // direction TO sun
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

    /// Draw the sky into the current pass (scene-color + opaque depth). Depth
    /// test is on (LEQUAL) with no depth write, and the sky is emitted at the
    /// far plane, so it fills only background pixels.
    pub fn render(self: *Skybox, camera: Camera3D, pass: PassSignature) void {
        const key = PipelineKey{
            .shader = self.shader,
            .layout = .fullscreen,
            .index_type = .u32,
            .indexed = false,
            .pass = pass,
            .primitive = .TRIANGLES,
            .cull = .NONE,
            .blend = .none,
            .depth_test = true, // occluded by nearer geometry
            .depth_write = false,
        };
        const pip = self.cache.get(key) catch |err| {
            std.log.err("Skybox: pipeline cache failed: {}", .{err});
            return;
        };

        const inv_vp = zm.inverse(camera.viewProjection());

        var params = SkyParams{
            .inv_view_proj = @bitCast(inv_vp),
            .camera_pos = .{ camera.position.x, camera.position.y, camera.position.z, 0 },
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
