//! src/render/ibl.zig
//!
//! Image-Based Lighting bake. Two stages:
//!   1. Render the environment (the skybox) into a cubemap (env_cube) — one
//!      fullscreen pass per face, reusing the sky shader.
//!   2. Convolve env_cube over the hemisphere into a small irradiance cubemap —
//!      the diffuse ambient term the lighting shaders will sample.
//!
//! Baking stage 1 from a cubemap (rather than sampling the sky procedurally in
//! the convolution) keeps the convolution source-agnostic: a loaded cubemap or
//! HDR-derived env_cube convolves identically later. Prefilter (specular) is the
//! next bake to live here.

const std = @import("std");
const sg = @import("sokol").gfx;
const zupra = @import("../root.zig");
const math = @import("../math.zig");
const zm = math.zm;
const pipeline = @import("../graphics/pipeline.zig");
const cubemap = @import("cubemap.zig");
const Cubemap = cubemap.Cubemap;
const FullscreenTriangle = @import("fullscreen.zig").FullscreenTriangle;
const Skybox = @import("skybox.zig").Skybox;

const shd_irr = @import("shaders").irradiance;

const PipelineCache = pipeline.PipelineCache;
const PipelineKey = pipeline.PipelineKey;
const PassSignature = pipeline.PassSignature;
const Matrix = math.Matrix;
const Vec3 = math.Vec3;

const env_size = 64; // env cubemap face size (source for convolution)
const irradiance_size = 32; // irradiance is low-frequency: small is fine

const ConvParams = extern struct {
    inv_view_proj: [16]f32,
};

pub const Ibl = struct {
    cache: *PipelineCache,
    env_cube: Cubemap,
    irradiance_cube: Cubemap,
    sampler: sg.Sampler, // linear, clamp on all axes
    irr_shader: sg.Shader,
    tri: FullscreenTriangle,

    pub fn init(cache: *PipelineCache) Ibl {
        return .{
            .cache = cache,
            .env_cube = Cubemap.initRenderTarget(env_size, .RGBA16F, 1),
            .irradiance_cube = Cubemap.initRenderTarget(irradiance_size, .RGBA16F, 1),
            .sampler = sg.makeSampler(.{
                .min_filter = .LINEAR,
                .mag_filter = .LINEAR,
                .wrap_u = .CLAMP_TO_EDGE,
                .wrap_v = .CLAMP_TO_EDGE,
                .wrap_w = .CLAMP_TO_EDGE,
            }),
            .irr_shader = sg.makeShader(shd_irr.irradianceShaderDesc(sg.queryBackend())),
            .tri = FullscreenTriangle.init(),
        };
    }

    pub fn deinit(self: *Ibl) void {
        self.tri.deinit();
        sg.destroyShader(self.irr_shader);
        sg.destroySampler(self.sampler);
        self.irradiance_cube.deinit();
        self.env_cube.deinit();
    }

    /// Run both bake stages. Call inside a frame (passes need a commit), once —
    /// the result is static unless the environment changes.
    pub fn bake(self: *Ibl, skybox: *Skybox) void {
        const face_vps = cubemap.faceViewProjections();
        const origin = Vec3{ .x = 0, .y = 0, .z = 0 };

        // Stage 1: sky -> env_cube (no depth, fill each face).
        const env_sig = faceSig(self.env_cube.format);
        for (0..6) |face| {
            const inv_vp = zm.inverse(face_vps[face]);
            zupra.beginDrawingPass(facePass(self.env_cube.faceAttachment(@intCast(face), 0)));
            skybox.renderRaw(inv_vp, origin, env_sig, false);
            zupra.endDrawing();
        }

        // Stage 2: env_cube -> irradiance_cube (convolution).
        const irr_sig = faceSig(self.irradiance_cube.format);
        for (0..6) |face| {
            const inv_vp = zm.inverse(face_vps[face]);
            zupra.beginDrawingPass(facePass(self.irradiance_cube.faceAttachment(@intCast(face), 0)));
            self.convolveFace(inv_vp, irr_sig);
            zupra.endDrawing();
        }
    }

    fn convolveFace(self: *Ibl, inv_vp: Matrix, sig: PassSignature) void {
        const key = PipelineKey{
            .shader = self.irr_shader,
            .layout = .fullscreen,
            .index_type = .u32,
            .indexed = false,
            .pass = sig,
            .primitive = .TRIANGLES,
            .cull = .NONE,
            .blend = .none,
            .depth_test = false,
            .depth_write = false,
        };
        const pip = self.cache.get(key) catch |err| {
            std.log.err("Ibl: convolve pipeline failed: {}", .{err});
            return;
        };

        var params = ConvParams{ .inv_view_proj = @bitCast(inv_vp) };

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = self.tri.vbuf;
        bindings.views[shd_irr.VIEW_env_cube] = self.env_cube.sample_view;
        bindings.samplers[shd_irr.SMP_smp] = self.sampler;

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(shd_irr.UB_conv_params, sg.asRange(&params));
        sg.draw(0, 3, 1);
    }

    /// The irradiance cubemap view + sampler, for the lighting shaders.
    pub fn irradianceView(self: Ibl) sg.View {
        return self.irradiance_cube.sample_view;
    }
    pub fn cubeSampler(self: Ibl) sg.Sampler {
        return self.sampler;
    }
};

/// Single-color, no-depth pass signature for a cube face of the given format.
fn faceSig(format: sg.PixelFormat) PassSignature {
    var sig = PassSignature{ .color_count = 1, .depth_format = .NONE, .sample_count = 1 };
    sig.color_formats[0] = format;
    return sig;
}

fn facePass(attachment: sg.View) sg.Pass {
    var att = sg.Attachments{};
    att.colors[0] = attachment;
    var action = sg.PassAction{};
    action.colors[0] = .{ .load_action = .CLEAR, .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 1 } };
    return .{ .action = action, .attachments = att };
}
