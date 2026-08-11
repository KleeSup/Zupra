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
//! The estimator is GTAO (see ssao.glsl): it finds exact horizon angles per
//! slice and solves the visibility integral across them in closed form, rather
//! than scattering sample points and counting hits. Same passes, same targets,
//! same consumers -- only the shader changed.
//!
//! Two passes: occlusion, then a depth-aware blur. The blur is not optional --
//! the occlusion pass deliberately trades banding for noise by jittering both
//! slice angle and march offset per pixel, and that trade only pays off if
//! something averages the noise back out.

const std = @import("std");
const sg = @import("sokol").gfx;
const zupra = @import("../root.zig");
const math = @import("../math.zig");
// zmath no longer needed here: the view-space formulation uses only scalars
// pulled out of the camera matrices, no inverse or full matrix multiply.
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
    view: [16]f32,
    proj_xy: [4]f32, // m00, m11, 1/m00, 1/m11
    depth_lin: [4]f32, // A, B, origin_top_left, unused
    params: [4]f32, // radius, intensity, thickness, falloff power
    bias: [4]f32, // angle bias (radians), unused x3
    counts: [4]f32, // slices, steps, target width, target height
    temporal: [4]f32, // rotation, offset, jitter enabled, unused
};

const BlurParams = extern struct {
    params: [4]f32, // step.x, step.y, radius, relative tolerance
    depth_lin: [4]f32, // A, B, unused, unused
};

pub const Settings = struct {
    /// Sampling radius in WORLD units: the size of the crevice you want
    /// darkened, not a quality dial. Too large and everything takes on a soft
    /// grey wash with no contact detail; too small and only the tightest
    /// corners register.
    ///
    /// The shader converts this to pixels and clamps the result to a usable
    /// range, so a distant surface still gets a real search width instead of a
    /// sub-pixel march that can never find a horizon -- which is what made a
    /// world-space-only radius silently produce an empty buffer at distance.
    radius: f32 = 1.0,
    /// Strength of the darkening. Above ~1.5 the result stops reading as
    /// occlusion and starts reading as dirt.
    intensity: f32 = 1.0,
    /// Vary the probe directions per frame so a temporal filter can converge
    /// them. Set this from whether TAA is actually running.
    ///
    /// ON WITHOUT TAA IT LOOKS WORSE, not better: the AO pattern changes every
    /// frame with nothing averaging it, so the image shimmers. With TAA it is
    /// the whole point -- successive frames probe different directions and the
    /// accumulation resolves detail no single frame contains, which is why
    /// shipping GTAO runs so few slices.
    temporal_jitter: bool = false,
    /// Slices through the hemisphere. Each is solved analytically, so this is
    /// the axis along which GTAO converges: 2 is usable, 3 is the common
    /// shipping choice, and beyond 4 the gain is hard to see.
    ///
    /// With temporal_jitter and TAA, 2 is generally enough -- the temporal
    /// accumulation supplies what the missing slices would have.
    ///
    /// No `bias` any more. Hemisphere sampling needed one because samples landed
    /// within depth precision of their own surface; horizon search compares
    /// angles rather than depths and has no equivalent failure.
    slices: u32 = 3,
    /// March steps per direction, so each slice costs 2x this many taps. The
    /// radius is divided across them, so more steps means finer occluders are
    /// found rather than a longer reach.
    steps: u32 = 4,
    /// Horizons within this many RADIANS of the tangent plane are ignored.
    ///
    /// Not a fudge factor -- it removes a specific statistical bias. Depth
    /// reconstruction scatters samples slightly either side of a flat surface,
    /// the horizon search takes max() over them, and the maximum of N noisy
    /// values drifts upward with N. Without this, a flat floor darkens in
    /// proportion to STEP COUNT.
    ///
    /// A real occluder rises well clear of the tangent, so a few degrees costs
    /// almost nothing in creases. Raise it if flat surfaces still darken as
    /// steps increase; lower it if shallow contact shadows disappear.
    /// 0.1 rad is about 6 degrees.
    angle_bias: f32 = 0.1,
    /// How aggressively a distant sample is discounted. A sample far behind the
    /// surface is usually a different object seen past an edge rather than a
    /// nearby wall, and counting it fully is what makes horizon methods draw
    /// dark halos around silhouettes. Raise it if you see them; lower it if
    /// occlusion feels too weak in deep corners.
    thickness: f32 = 1.0,
    /// Exponent on the final term. Above 1 deepens contact shadows while
    /// leaving open areas alone; 1.0 is linear.
    falloff: f32 = 1.0,
    /// Blur kernel radius in taps per axis (max 8). The blur is separable, so
    /// cost is linear in this, not quadratic.
    blur_radius: u32 = 3,
    /// How far a neighbour's depth may differ before the blur rejects it, as a
    /// FRACTION of the centre pixel's view depth. Relative rather than absolute
    /// so the same value behaves the same at one metre and at fifty.
    blur_tolerance: f32 = 0.02,
    /// Render AO at half resolution. The view-space formulation is cheap enough
    /// to run at full resolution on a desktop GPU; half is the setting for
    /// weaker hardware, and it costs edge definition on small features.
    half_resolution: bool = false,
};

pub const Ssao = struct {
    cache: *PipelineCache,
    shader: sg.Shader,
    blur_shader: sg.Shader,
    tri: FullscreenTriangle,
    /// POINT sampling for everything these passes read.
    ///
    /// Depth and normals must never be interpolated. Across a silhouette,
    /// linear filtering blends the near surface's depth with the far one's and
    /// returns a value belonging to neither -- a phantom surface hanging in the
    /// gap. The occlusion pass reconstructs a position there, its hemisphere
    /// straddles both real surfaces, and the result is a one-pixel dark line
    /// tracing every edge that swims as geometry moves across the pixel grid.
    /// DeferredRenderer samples the G-buffer with NEAREST for the same reason.
    point_sampler: sg.Sampler,
    /// Linear, for whoever upsamples the finished buffer. Only matters when
    /// half_resolution is on; at full resolution the consumer should point
    /// sample this too.
    sampler: sg.Sampler,

    /// Ping and pong: occlusion is written to `ao`, blurred into `blurred`.
    ao: Framebuffer,
    blurred: Framebuffer,

    settings: Settings = .{},
    /// Ticks once per render, driving the temporal jitter sequence.
    frame: u32 = 0,
    width: u32 = 0,
    height: u32 = 0,

    pub fn init(cache: *PipelineCache, width: u32, height: u32, settings: Settings) Ssao {
        const dims = scaled(width, height, settings);
        return .{
            .cache = cache,
            .shader = sg.makeShader(shd.ssaoShaderDesc(sg.queryBackend())),
            .blur_shader = sg.makeShader(shd_blur.ssaoBlurShaderDesc(sg.queryBackend())),
            .tri = FullscreenTriangle.init(),
            .point_sampler = sg.makeSampler(.{
                .min_filter = .NEAREST,
                .mag_filter = .NEAREST,
                .wrap_u = .CLAMP_TO_EDGE,
                .wrap_v = .CLAMP_TO_EDGE,
            }),
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
        sg.destroySampler(self.point_sampler);
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

    /// Run all three passes (occlusion, then blur across each axis). Call after
    /// the G-buffer pass and BEFORE deferred lighting, which samples the result.
    pub fn render(self: *Ssao, camera: Camera3D, depth_view: sg.View, normal_view: sg.View) void {
        const inv_w = 1.0 / @as(f32, @floatFromInt(self.width));
        const inv_h = 1.0 / @as(f32, @floatFromInt(self.height));

        // Depth linearization constants. zmath's perspectiveFovLh puts
        // A = f/(f-n) and B = -n*f/(f-n) in the matrix, giving
        // depth = A + B/z and therefore z = B/(depth - A).
        const n = camera.near;
        const f = camera.far;
        const a = f / (f - n);
        const b = -n * f / (f - n);
        const depth_lin = [4]f32{ a, b, 0, 0 };

        // The projection's x and y scales are all the inner loop needs to move
        // between view space and the screen -- no matrix required.
        const proj = camera.projection();
        const m00 = proj[0][0];
        const m11 = proj[1][1];

        // Pass 1: occlusion.
        zupra.beginDrawingFramebuffer(self.ao);
        {
            const jitter = temporalJitter(self.frame);

            var params = SsaoParams{
                .view = @bitCast(camera.view()),
                .proj_xy = .{ m00, m11, 1.0 / m00, 1.0 / m11 },
                .depth_lin = .{
                    a,
                    b,
                    if (sg.queryFeatures().origin_top_left) 1.0 else 0.0,
                    0,
                },
                .params = .{
                    self.settings.radius,
                    self.settings.intensity,
                    self.settings.thickness,
                    self.settings.falloff,
                },
                .bias = .{ self.settings.angle_bias, 0, 0, 0 },

                .temporal = .{
                    jitter.rotation,
                    jitter.offset,
                    if (self.settings.temporal_jitter) 1.0 else 0.0,
                    0,
                },

                .counts = .{
                    @floatFromInt(@max(1, self.settings.slices)),
                    @floatFromInt(@max(1, self.settings.steps)),
                    @floatFromInt(self.width),
                    @floatFromInt(self.height),
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
            bindings.samplers[shd.SMP_smp] = self.point_sampler;

            sg.applyPipeline(pip);
            sg.applyBindings(bindings);
            sg.applyUniforms(shd.UB_ssao_params, sg.asRange(&params));
            sg.draw(0, 3, 1);
        }
        zupra.endDrawing();

        // The jitter sequences are 6 and 4 long, so 12 covers both without the
        // counter ever growing large enough to lose precision as a float.
        self.frame +%= 1;

        // Passes 2 and 3: separable blur. Horizontal ao -> blurred, then
        // vertical blurred -> ao, so the finished result lands back in `ao` and
        // no third target is needed.
        const r: f32 = @floatFromInt(@min(self.settings.blur_radius, 8));
        self.blurPass(self.ao, self.blurred, depth_view, .{ inv_w, 0 }, r, depth_lin);
        self.blurPass(self.blurred, self.ao, depth_view, .{ 0, inv_h }, r, depth_lin);
    }

    fn blurPass(
        self: *Ssao,
        src: Framebuffer,
        dst: Framebuffer,
        depth_view: sg.View,
        step: [2]f32,
        radius: f32,
        depth_lin: [4]f32,
    ) void {
        zupra.beginDrawingFramebuffer(dst);
        defer zupra.endDrawing();

        const pip = self.pipelineFor(self.blur_shader, dst.passSignature()) orelse return;

        var params = BlurParams{
            .params = .{ step[0], step[1], radius, self.settings.blur_tolerance },
            .depth_lin = depth_lin,
        };

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = self.tri.vbuf;
        bindings.views[shd_blur.VIEW_tex_ao] = src.sample_view;
        bindings.views[shd_blur.VIEW_tex_depth] = depth_view;
        bindings.samplers[shd_blur.SMP_smp] = self.point_sampler;

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(shd_blur.UB_blur_params, sg.asRange(&params));
        sg.draw(0, 3, 1);
    }

    /// The finished occlusion buffer, for the lighting pass to multiply in.
    ///
    /// `sample_view`, NOT `color_view`. A Framebuffer carries both: color_view
    /// is the attachment it renders INTO, sample_view is the texture read back
    /// out. Binding the attachment into a sampler slot is what sokol reports as
    /// VALIDATE_ABND_EXPECT_TEXVIEW.
    pub fn aoView(self: Ssao) sg.View {
        return self.ao.sample_view;
    }

    /// The AO buffer as a Texture, for drawing it to the screen through the
    /// SpriteBatch. Seeing the raw buffer is worth far more than inferring its
    /// contents from how the lit scene looks -- an all-black buffer and a
    /// correct-but-subtle one produce very different images, and only one of
    /// them is a bug.
    pub fn debugTexture(self: Ssao) tex.Texture {
        return self.ao.asTexture();
    }
    /// Sampler for the lighting pass. Linear only helps when the buffer is at
    /// half resolution and needs upsampling; at full resolution either is fine.
    pub fn aoSampler(self: Ssao) sg.Sampler {
        return if (self.settings.half_resolution) self.sampler else self.point_sampler;
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

fn temporalJitter(frame: u32) struct { rotation: f32, offset: f32 } {
    const i6_ = frame % 6;
    const i4_ = frame % 4;

    const rotations = [_]f32{
        1.0 / 6.0,
        5.0 / 6.0,
        3.0 / 6.0,
        4.0 / 6.0,
        2.0 / 6.0,
        0.0,
    };

    const offsets = [_]f32{
        0.0,
        0.5,
        0.25,
        0.75,
    };

    return .{
        .rotation = rotations[i6_],
        .offset = offsets[i4_],
    };
}
