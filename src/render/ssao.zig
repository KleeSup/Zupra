//! src/render/ssao.zig
//!
//! Screen-space ambient occlusion.
//!
//! Fills the one large gap left in the lighting model: image-based lighting
//! arrives completely unoccluded. Every surface receives the full hemisphere of
//! sky regardless of what is standing next to it, so ambient light reaches under
//! objects and into corners at full strength and geometry reads as floating.
//!
//! A shadow map cannot address this. A shadow map answers "what can this POINT
//! see", and the sky is an area source covering the whole hemisphere -- there is
//! no single view to render depth from. Every engine approximates the answer
//! instead, and screen-space occlusion is the cheapest useful approximation.
//!
//! DEFERRED ONLY, for now. The G-buffer already carries world-space normals and
//! depth, which is exactly the input required. The forward path has depth (from
//! the prepass) but no normals; giving it AO means either a normals target or
//! reconstructing normals from depth derivatives, which is a separate piece of
//! work and noticeably worse at silhouettes.
//!
//! Two passes: occlusion, then a depth-aware blur. The blur is not optional --
//! the occlusion pass deliberately trades banding for noise by rotating its
//! sample directions per pixel, and that trade only pays off if something
//! averages the noise back out.

const std = @import("std");
const sg = @import("sokol").gfx;
const zupra = @import("../root.zig");
const math = @import("../math.zig");
const zm = math.zm;
const pipeline = @import("../graphics/pipeline.zig");
const FullscreenTriangle = @import("fullscreen.zig").FullscreenTriangle;
const Framebuffer = @import("framebuffer.zig").Framebuffer;
const tex = @import("../graphics/texture.zig");
const Camera3D = @import("camera3d.zig").Camera3D;

const shd = @import("shaders").ssao;
const shd_blur = @import("shaders").ssao_blur;

const PipelineCache = pipeline.PipelineCache;
const PipelineKey = pipeline.PipelineKey;
const PassSignature = pipeline.PassSignature;
const Matrix = math.Matrix;

const SsaoParams = extern struct {
    inv_view_proj: [16]f32,
    view_proj: [16]f32,
    camera_pos: [4]f32,
    params: [4]f32, // radius, intensity, bias, origin_top_left
    tuning: [4]f32, // sample count, falloff power, 1/w, 1/h
};

const BlurParams = extern struct {
    params: [4]f32, // 1/w, 1/h, depth sharpness, kernel radius
};

pub const Settings = struct {
    /// Sampling radius in WORLD units. This is the scale of the detail AO can
    /// resolve: it is the size of the crevice you want darkened, not a quality
    /// dial. Too large and everything picks up a soft grey wash with no
    /// contact detail; too small and only the tightest corners register.
    radius: f32 = 0.6,
    /// Strength of the darkening. Above ~1.5 the result stops reading as
    /// occlusion and starts reading as dirt.
    intensity: f32 = 1.0,
    /// Depth bias in world units, against self-occlusion on flat surfaces where
    /// samples land within depth precision of the surface they came from.
    bias: f32 = 0.02,
    /// Samples per pixel, capped at 32 by the shader loop. 16 is the usual
    /// sweet spot; the cost is close to linear in this.
    samples: u32 = 16,
    /// Exponent on the final term. Above 1 deepens contact shadows while
    /// leaving open areas alone; 1.0 is linear.
    falloff: f32 = 1.5,
    /// Blur kernel radius in texels (max 4). Larger smooths more noise and
    /// costs quadratically.
    blur_radius: u32 = 2,
    /// How sharply the blur rejects taps at differing depth. Higher confines
    /// the filter more tightly to one surface, at the cost of leaving more
    /// noise near edges.
    blur_sharpness: f32 = 400.0,
    /// Render AO at half resolution. AO is low-frequency, so the quality cost
    /// is small and the saving is roughly four times the pass.
    half_resolution: bool = true,
};

pub const Ssao = struct {
    cache: *PipelineCache,
    shader: sg.Shader,
    blur_shader: sg.Shader,
    tri: FullscreenTriangle,
    sampler: sg.Sampler,

    /// Ping and pong: occlusion is written to `ao`, blurred into `blurred`.
    ao: Framebuffer,
    blurred: Framebuffer,

    settings: Settings = .{},
    width: u32 = 0,
    height: u32 = 0,

    pub fn init(cache: *PipelineCache, width: u32, height: u32, settings: Settings) Ssao {
        const dims = scaled(width, height, settings);
        return .{
            .cache = cache,
            .shader = sg.makeShader(shd.ssaoShaderDesc(sg.queryBackend())),
            .blur_shader = sg.makeShader(shd_blur.ssaoBlurShaderDesc(sg.queryBackend())),
            .tri = FullscreenTriangle.init(),
            // CLAMP, and LINEAR so a half-resolution buffer upsamples smoothly
            // when the lighting pass reads it at full resolution.
            .sampler = sg.makeSampler(.{
                .min_filter = .LINEAR,
                .mag_filter = .LINEAR,
                .wrap_u = .CLAMP_TO_EDGE,
                .wrap_v = .CLAMP_TO_EDGE,
            }),
            // R8: a single visibility scalar in [0,1]. A wider format would buy
            // precision the blur immediately discards. depth_format .NONE --
            // both passes are fullscreen with no depth test, and allocating a
            // depth buffer per AO target would be pure waste.
            .ao = makeTarget(dims.w, dims.h),
            .blurred = makeTarget(dims.w, dims.h),
            .settings = settings,
            .width = dims.w,
            .height = dims.h,
        };
    }

    pub fn deinit(self: *Ssao) void {
        self.blurred.deinit();
        self.ao.deinit();
        self.tri.deinit();
        sg.destroySampler(self.sampler);
        sg.destroyShader(self.blur_shader);
        sg.destroyShader(self.shader);
    }

    pub fn resize(self: *Ssao, width: u32, height: u32) void {
        const dims = scaled(width, height, self.settings);
        if (dims.w == self.width and dims.h == self.height) return;
        self.rebuild(width, height);
    }

    /// Reallocate unconditionally. resize() early-outs when the dimensions match,
    /// which is wrong after `half_resolution` changes: the window is the same
    /// size but the targets should not be.
    pub fn rebuild(self: *Ssao, width: u32, height: u32) void {
        const dims = scaled(width, height, self.settings);
        self.ao.deinit();
        self.blurred.deinit();
        self.ao = makeTarget(dims.w, dims.h);
        self.blurred = makeTarget(dims.w, dims.h);
        self.width = dims.w;
        self.height = dims.h;
    }

    /// Run both passes. Call after the G-buffer pass and BEFORE deferred
    /// lighting, which samples the result.
    pub fn render(self: *Ssao, camera: Camera3D, depth_view: sg.View, normal_view: sg.View) void {
        const vp = camera.viewProjection();
        const inv_w = 1.0 / @as(f32, @floatFromInt(self.width));
        const inv_h = 1.0 / @as(f32, @floatFromInt(self.height));

        // Pass 1: occlusion.
        zupra.beginDrawingFramebuffer(self.ao);
        {
            var params = SsaoParams{
                .inv_view_proj = @bitCast(zm.inverse(vp)),
                .view_proj = @bitCast(vp),
                .camera_pos = .{ camera.position.x, camera.position.y, camera.position.z, 0 },
                .params = .{
                    self.settings.radius,
                    self.settings.intensity,
                    self.settings.bias,
                    // Same flag the shadow sampler uses: on backends whose
                    // render targets read top-left first, screen UVs and NDC y
                    // run opposite ways, and this pass converts between them
                    // twice per sample.
                    if (sg.queryFeatures().origin_top_left) 1.0 else 0.0,
                },
                .tuning = .{
                    @floatFromInt(@min(self.settings.samples, 32)),
                    self.settings.falloff,
                    inv_w,
                    inv_h,
                },
            };

            const pip = self.pipelineFor(self.shader, self.ao.passSignature()) orelse {
                zupra.endDrawing();
                return;
            };

            var bindings = sg.Bindings{};
            bindings.vertex_buffers[0] = self.tri.vbuf;
            bindings.views[shd.VIEW_tex_depth] = depth_view;
            bindings.views[shd.VIEW_tex_normal] = normal_view;
            bindings.samplers[shd.SMP_smp] = self.sampler;

            sg.applyPipeline(pip);
            sg.applyBindings(bindings);
            sg.applyUniforms(shd.UB_ssao_params, sg.asRange(&params));
            sg.draw(0, 3, 1);
        }
        zupra.endDrawing();

        // Pass 2: depth-aware blur.
        zupra.beginDrawingFramebuffer(self.blurred);
        {
            var params = BlurParams{ .params = .{
                inv_w,
                inv_h,
                self.settings.blur_sharpness,
                @floatFromInt(@min(self.settings.blur_radius, 4)),
            } };

            const pip = self.pipelineFor(self.blur_shader, self.blurred.passSignature()) orelse {
                zupra.endDrawing();
                return;
            };

            var bindings = sg.Bindings{};
            bindings.vertex_buffers[0] = self.tri.vbuf;
            bindings.views[shd_blur.VIEW_tex_ao] = self.ao.sample_view;
            bindings.views[shd_blur.VIEW_tex_depth] = depth_view;
            bindings.samplers[shd_blur.SMP_smp] = self.sampler;

            sg.applyPipeline(pip);
            sg.applyBindings(bindings);
            sg.applyUniforms(shd_blur.UB_blur_params, sg.asRange(&params));
            sg.draw(0, 3, 1);
        }
        zupra.endDrawing();
    }

    /// The finished occlusion buffer, for the lighting pass to multiply in.
    ///
    /// `sample_view`, NOT `color_view`. A Framebuffer carries both: color_view
    /// is the attachment it renders INTO, sample_view is the texture read back
    /// out. Binding the attachment into a sampler slot is what sokol reports as
    /// VALIDATE_ABND_EXPECT_TEXVIEW.
    pub fn aoView(self: Ssao) sg.View {
        return self.blurred.sample_view;
    }

    /// The AO buffer as a Texture, for drawing it to the screen through the
    /// SpriteBatch. Seeing the raw buffer is worth far more than inferring its
    /// contents from how the lit scene looks -- an all-black buffer and a
    /// correct-but-subtle one produce very different images, and only one of
    /// them is a bug.
    pub fn debugTexture(self: Ssao) tex.Texture {
        return self.blurred.asTexture();
    }
    pub fn aoSampler(self: Ssao) sg.Sampler {
        return self.sampler;
    }

    fn pipelineFor(self: *Ssao, shader: sg.Shader, sig: PassSignature) ?sg.Pipeline {
        return self.cache.get(.{
            .shader = shader,
            .layout = .fullscreen,
            .index_type = .u32,
            .indexed = false,
            .pass = sig,
            .primitive = .TRIANGLES,
            .cull = .NONE,
            .blend = .none,
            .depth_test = false,
            .depth_write = false,
        }) catch |err| {
            std.log.err("Ssao: pipeline cache failed: {}", .{err});
            return null;
        };
    }
};

fn makeTarget(w: u32, h: u32) Framebuffer {
    return Framebuffer.init(.{
        .width = w,
        .height = h,
        .color_format = .R8,
        .depth_format = .NONE,
    });
}

fn scaled(width: u32, height: u32, s: Settings) struct { w: u32, h: u32 } {
    const div: u32 = if (s.half_resolution) 2 else 1;
    return .{ .w = @max(1, width / div), .h = @max(1, height / div) };
}
