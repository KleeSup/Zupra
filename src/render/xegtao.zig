//! src/render/xegtao.zig
//!
//! Ambient occlusion, ported from the technique in Intel's XeGTAO.
//!
//! Replaces the hand-rolled GTAO. That version failed repeatedly in the same
//! place: not in the visibility integral, which is well defined and was correct,
//! but in the decisions surrounding it. How far to search in screen space, what
//! to do when that distance is a fraction of a pixel or several hundred, which
//! samples across a depth discontinuity are real occluders. Each wrong answer
//! produced an artifact that looked like a different bug.
//!
//! XeGTAO answers all of those, and the central one is the depth pyramid. A
//! horizon search wants to cover a world radius, which projects to a wildly
//! varying pixel count. Marching in pixels and clamping the range makes the
//! measured volume depend on where the camera stands. Marching in world units
//! costs an unbounded tap count up close. Reading a coarser pyramid level for
//! more distant samples removes the dilemma: a fixed, small tap count covers the
//! full radius from any distance, so the measurement stops depending on the
//! viewpoint.
//!
//! PASS SEQUENCE
//!   1. Prefilter level 0: hardware depth to view-space Z.
//!   2. Prefilter levels 1..N: depth-aware downsample, weighted toward the
//!      farthest of each four so thin near geometry cannot become a phantom
//!      occluder at coarse scale.
//!   3. Main pass: horizon search against the pyramid.
//!   4. Denoise: separable, depth-aware, one pass per axis.
//!
//! Deferred and forward both work, since both now produce depth and world-space
//! normals. Same interface as the module it replaces, so nothing downstream
//! changes.

const std = @import("std");
const sg = @import("sokol").gfx;
const zupra = @import("../root.zig");
const math = @import("../math.zig");
const pipeline = @import("../graphics/pipeline.zig");
const gfx = @import("../graphics/graphics.zig");
const fb = @import("framebuffer.zig");
const Camera3D = @import("camera3d.zig").Camera3D;

const shd_pre = @import("shaders").xegtao_prefilter;
const shd_main = @import("shaders").xegtao_main;
const shd_den = @import("shaders").xegtao_denoise;

const PipelineCache = pipeline.PipelineCache;
const PipelineKey = pipeline.PipelineKey;
const PassSignature = pipeline.PassSignature;
const Framebuffer = fb.Framebuffer;
const Texture = gfx.texture.Texture;
const Vertex2D = gfx.Vertex2D;
const Matrix = math.Matrix;

/// Levels in the depth pyramid. Five reaches a sixteenth of the screen, which is
/// far enough for any radius worth searching; beyond that a level is small enough
/// that its texels describe the scene too coarsely to be useful.
pub const mip_levels: u32 = 5;

const PrefilterParams = extern struct {
    params: [4]f32, // depth A, depth B, is_level_zero, unused
    source: [4]f32, // 1/src width, 1/src height, src mip, unused
    falloff: [4]f32, // falloff mul, falloff add, unused, unused
};

const MainParams = extern struct {
    view: [16]f32,
    ndc_to_view_mul: [4]f32,
    ndc_to_view_add: [4]f32,
    viewport: [4]f32, // 1/w, 1/h, w, h
    radius_params: [4]f32, // radius, falloff range, distribution power, thin comp
    counts: [4]f32, // slices, steps, final power, mip sampling offset
    temporal: [4]f32, // frame, jitter on, unused, unused
};

const DenoiseParams = extern struct {
    params: [4]f32, // step x, step y, radius, tolerance
};

pub const Settings = struct {
    /// Search radius in world units. The size of the crevice you want darkened,
    /// not a quality dial.
    ///
    /// Unlike the previous implementation this means the same thing from every
    /// distance, because the pyramid removes the need to clamp it in pixels.
    radius: f32 = 0.5,
    /// Fraction of the radius over which a sample's influence fades to nothing.
    /// The outer part of the search contributes progressively less, so there is
    /// no hard boundary where occlusion stops.
    falloff_range: f32 = 0.615,
    /// Exponent on the sample distribution. Above 1 concentrates samples near
    /// the pixel, which is where occlusion detail is. 2.0 is the reference
    /// default and rarely wants changing.
    distribution_power: f32 = 2.0,
    /// How aggressively a sample behind the surface is discounted.
    ///
    /// This is the test the previous implementation never had. It only measured
    /// how FAR a sample was, never which side of a depth discontinuity it sat
    /// on, so geometry seen through a window or past a panel edge became a
    /// strong occluder and painted regions shaped like whatever lay behind the
    /// gap. Raise it if that reappears; lower it if thin objects stop occluding.
    /// The reference default. Raise toward 0.7 if geometry seen through gaps
    /// starts occluding; it stretches a sample's depth delta before the falloff
    /// measures it, so far-side samples fall out of range naturally.
    thin_occluder_compensation: f32 = 0.0,
    /// Exponent on the final visibility. Above 1 deepens contact darkening while
    /// leaving open areas alone.
    final_power: f32 = 1.0,
    /// Slices through the hemisphere. Each is solved analytically, so this is
    /// the convergence axis. Two is the reference default with temporal
    /// accumulation, three without.
    slices: u32 = 3,
    /// Steps per slice, each covering both directions. The pyramid means more
    /// steps buy finer detail rather than a longer reach.
    steps: u32 = 3,
    /// Bias on the mip chosen per sample. Negative reads finer levels, which is
    /// sharper and slower; positive reads coarser, which is faster and blurrier.
    mip_offset: f32 = 3.3,
    /// Vary probe directions per frame so a temporal filter can converge them.
    /// On without TAA this shimmers rather than converging, so it should follow
    /// whether TAA is actually running.
    temporal_jitter: bool = false,
    /// Denoise kernel radius in taps per axis, maximum 8. Separable, so cost is
    /// linear in this.
    denoise_radius: u32 = 2,
    /// How far a neighbour's depth may differ before the denoiser rejects it, as
    /// a fraction of the centre pixel's view depth.
    denoise_tolerance: f32 = 0.02,
    /// Run at half resolution. The pyramid already bounds the cost, so full
    /// resolution is affordable on a desktop GPU and is the better default.
    half_resolution: bool = false,
};

pub const XeGtao = struct {
    cache: *PipelineCache,
    prefilter_shader: sg.Shader,
    main_shader: sg.Shader,
    denoise_shader: sg.Shader,
    sampler: sg.Sampler,
    vbuf: sg.Buffer,

    /// View-space depth pyramid, split across TWO images by level parity.
    ///
    /// One image would be natural, each level downsampled from the one above.
    /// Sokol rejects it: the check compares IMAGES, not mip ranges, so reading
    /// any level of an image while any level of it is attached fails regardless
    /// of whether they overlap (sokol_gfx.h: "the same image cannot be used as
    /// texture binding and pass attachment").
    ///
    /// So even levels live in image 0 and odd levels in image 1. Writing level N
    /// reads level N-1, which is always in the other image. Both carry the full
    /// mip chain and each simply leaves the levels it does not own unwritten,
    /// which costs one extra single-channel image.
    ///
    /// The main pass then binds both and selects by parity, so nothing has to be
    /// gathered afterwards.
    depth_img: [2]sg.Image = .{ .{}, .{} },
    /// Attachment view per level, into whichever image owns that level.
    depth_attach: [mip_levels]sg.View = undefined,
    /// Single-level texture view per level, for the downsample that reads it.
    depth_read: [mip_levels]sg.View = undefined,
    /// Full-chain texture view per image, for the main pass and the denoiser.
    depth_texture: [2]sg.View = .{ .{}, .{} },
    mip_width: [mip_levels]u32 = undefined,
    mip_height: [mip_levels]u32 = undefined,
    /// Levels actually created. Fewer than mip_levels at small window sizes,
    /// since a level below one texel is meaningless.
    live_mips: u32 = 0,

    /// Occlusion, and the denoiser's second target. Ping-ponged so the finished
    /// result lands back in `ao`.
    ao: Framebuffer = undefined,
    scratch: Framebuffer = undefined,

    settings: Settings = .{},
    frame: u64 = 0,
    width: u32 = 0,
    height: u32 = 0,

    pub fn init(cache: *PipelineCache, width: u32, height: u32, settings: Settings) XeGtao {
        const top_left = sg.queryFeatures().origin_top_left;
        const clip = [3][2]f32{ .{ -1, -1 }, .{ 3, -1 }, .{ -1, 3 } };
        var verts: [3]Vertex2D = undefined;
        for (clip, 0..) |p, i| {
            const u = 0.5 + 0.5 * p[0];
            const v = if (top_left) 0.5 - 0.5 * p[1] else 0.5 + 0.5 * p[1];
            verts[i] = .{ .pos = p, .uv = .{ u, v }, .color = 0xFFFFFFFF };
        }

        var self = XeGtao{
            .cache = cache,
            .prefilter_shader = sg.makeShader(shd_pre.xegtaoPrefilterShaderDesc(sg.queryBackend())),
            .main_shader = sg.makeShader(shd_main.xegtaoMainShaderDesc(sg.queryBackend())),
            .denoise_shader = sg.makeShader(shd_den.xegtaoDenoiseShaderDesc(sg.queryBackend())),
            // LINEAR with mipmaps, because the main pass selects a mip level per
            // sample and needs to read it. NEAREST within a level would be more
            // correct at silhouettes, but the prefilter's outlier rejection has
            // already removed the phantom surfaces that motivated point sampling.
            .sampler = sg.makeSampler(.{
                .min_filter = .LINEAR,
                .mag_filter = .LINEAR,
                .mipmap_filter = .NEAREST,
                .wrap_u = .CLAMP_TO_EDGE,
                .wrap_v = .CLAMP_TO_EDGE,
            }),
            .vbuf = sg.makeBuffer(.{ .data = sg.asRange(&verts) }),
            .settings = settings,
        };
        self.build(width, height);
        return self;
    }

    pub fn deinit(self: *XeGtao) void {
        self.destroyTargets();
        sg.destroyBuffer(self.vbuf);
        sg.destroySampler(self.sampler);
        sg.destroyShader(self.denoise_shader);
        sg.destroyShader(self.main_shader);
        sg.destroyShader(self.prefilter_shader);
    }

    pub fn resize(self: *XeGtao, width: u32, height: u32) void {
        const dims = scaled(width, height, self.settings);
        if (dims.w == self.width and dims.h == self.height) return;
        self.destroyTargets();
        self.build(width, height);
    }

    /// Reallocate unconditionally, for when a setting that affects the targets
    /// changed but the window size did not.
    pub fn rebuild(self: *XeGtao, width: u32, height: u32) void {
        self.destroyTargets();
        self.build(width, height);
    }

    fn destroyTargets(self: *XeGtao) void {
        if (self.width == 0) return;
        var i: u32 = 0;
        while (i < self.live_mips) : (i += 1) {
            sg.destroyView(self.depth_attach[i]);
            sg.destroyView(self.depth_read[i]);
        }
        for (self.depth_texture) |v| if (v.id != 0) sg.destroyView(v);
        for (self.depth_img) |img| if (img.id != 0) sg.destroyImage(img);
        self.ao.deinit();
        self.scratch.deinit();
        self.live_mips = 0;
        self.width = 0;
    }

    fn build(self: *XeGtao, width: u32, height: u32) void {
        const dims = scaled(width, height, self.settings);
        self.width = dims.w;
        self.height = dims.h;

        // How many levels this resolution can actually support.
        var count: u32 = 0;
        var w = dims.w;
        var h = dims.h;
        while (count < mip_levels and w >= 1 and h >= 1) : (count += 1) {
            self.mip_width[count] = w;
            self.mip_height[count] = h;
            w = @max(1, w / 2);
            h = @max(1, h / 2);
        }
        self.live_mips = @max(1, count);

        // R32F, not R16F. This holds view-space Z in world units, and half
        // precision runs out well before the far plane: at 1000 units the gap
        // between representable values is larger than most geometry. Precision
        // problems here surface as AO that fails only at distance, which is
        // exactly the kind of bug that costs days.
        var pair: u32 = 0;
        while (pair < 2) : (pair += 1) {
            self.depth_img[pair] = sg.makeImage(.{
                .width = @intCast(dims.w),
                .height = @intCast(dims.h),
                .num_mipmaps = @intCast(self.live_mips),
                .pixel_format = .R32F,
                .usage = .{ .color_attachment = true },
            });
            self.depth_texture[pair] = sg.makeView(.{
                .texture = .{
                    .image = self.depth_img[pair],
                    .mip_levels = .{ .base = 0, .count = @intCast(self.live_mips) },
                },
            });
        }

        var i: u32 = 0;
        while (i < self.live_mips) : (i += 1) {
            const owner = self.depth_img[i & 1];
            self.depth_attach[i] = sg.makeView(.{
                .color_attachment = .{ .image = owner, .mip_level = @intCast(i) },
            });
            self.depth_read[i] = sg.makeView(.{
                .texture = .{
                    .image = owner,
                    .mip_levels = .{ .base = @intCast(i), .count = 1 },
                },
            });
        }

        self.ao = makeTarget(dims.w, dims.h);
        self.scratch = makeTarget(dims.w, dims.h);
    }

    /// Run every pass. Call after the geometry pass and before lighting, which
    /// samples the result.
    pub fn render(self: *XeGtao, camera: Camera3D, depth_view: sg.View, normal_view: sg.View) void {
        // Depth linearization constants. perspectiveFovLh puts A = f/(f-n) and
        // B = -n*f/(f-n) in the matrix, giving depth = A + B/z and therefore
        // z = B/(depth - A).
        const n = camera.near;
        const f = camera.far;
        const a = f / (f - n);
        const b = -n * f / (f - n);
        //const top_left: f32 = if (sg.queryFeatures().origin_top_left) 1.0 else 0.0;

        // The prefilter's outlier rejection uses the same falloff shape as the
        // main pass, scaled down: a coarse level should reject a near sliver
        // more readily than the search itself does.
        const filter_radius = 0.75 * self.settings.radius;
        const filter_range = @max(self.settings.falloff_range * filter_radius, 1e-4);
        const filter_from = filter_radius * (1.0 - self.settings.falloff_range);
        const filter_mul = -1.0 / filter_range;
        const filter_add = filter_from / filter_range + 1.0;

        // Pass 1: hardware depth to view-space Z, into level 0.
        {
            var p = PrefilterParams{
                .params = .{ a, b, 1.0, 0 },
                .source = .{
                    1.0 / @as(f32, @floatFromInt(self.width)),
                    1.0 / @as(f32, @floatFromInt(self.height)),
                    0,
                    0,
                },
                .falloff = .{ filter_mul, filter_add, 0, 0 },
            };
            self.prefilterPass(0, depth_view, &p);
        }

        // Passes 2..N: depth-aware downsample of the level above.
        var level: u32 = 1;
        while (level < self.live_mips) : (level += 1) {
            var p = PrefilterParams{
                .params = .{ a, b, 0.0, 0 },
                .source = .{
                    1.0 / @as(f32, @floatFromInt(self.mip_width[level - 1])),
                    1.0 / @as(f32, @floatFromInt(self.mip_height[level - 1])),
                    // Zero, not level-1: the bound view spans one level, so
                    // within it the source is always mip 0.
                    0,
                    0,
                },
                .falloff = .{ filter_mul, filter_add, 0, 0 },
            };
            self.prefilterPass(level, self.depth_read[level - 1], &p);
        }

        // Pass 3: the horizon search.
        {
            const proj = camera.projection();
            const tan_half_x = 1.0 / proj[0][0];
            const tan_half_y = 1.0 / proj[1][1];

            // Screen uv straight to a view ray, as the reference does, rather
            // than converting to NDC and flipping separately. The y terms carry
            // the flip in their sign, so the conversion happens exactly once and
            // there is no second place for an origin correction to be applied by
            // mistake.
            //
            // On a backend whose targets read bottom-up the v axis already runs
            // the other way, so the y signs invert with it.
            const flip = sg.queryFeatures().origin_top_left;
            const y_mul: f32 = if (flip) -2.0 * tan_half_y else 2.0 * tan_half_y;
            const y_add: f32 = if (flip) tan_half_y else -tan_half_y;

            var p = MainParams{
                .view = @bitCast(camera.view()),
                .ndc_to_view_mul = .{ 2.0 * tan_half_x, y_mul, 0, 0 },
                .ndc_to_view_add = .{ -tan_half_x, y_add, 0, 0 },
                .viewport = .{
                    1.0 / @as(f32, @floatFromInt(self.width)),
                    1.0 / @as(f32, @floatFromInt(self.height)),
                    @floatFromInt(self.width),
                    @floatFromInt(self.height),
                },
                .radius_params = .{
                    self.settings.radius,
                    self.settings.falloff_range,
                    self.settings.distribution_power,
                    self.settings.thin_occluder_compensation,
                },
                .counts = .{
                    @floatFromInt(@max(1, self.settings.slices)),
                    @floatFromInt(@max(1, self.settings.steps)),
                    self.settings.final_power,
                    self.settings.mip_offset,
                },
                .temporal = .{
                    @floatFromInt(self.frame % 64),
                    if (self.settings.temporal_jitter) 1.0 else 0.0,
                    0,
                    0,
                },
            };

            zupra.beginDrawingFramebufferClear(self.ao, .{ .r = 1, .g = 1, .b = 1, .a = 1 });
            defer zupra.endDrawing();

            const pip = self.pipelineFor(self.main_shader, self.ao.passSignature()) orelse return;
            var bindings = sg.Bindings{};
            bindings.vertex_buffers[0] = self.vbuf;
            bindings.views[shd_main.VIEW_tex_depth_even] = self.depth_texture[0];
            bindings.views[shd_main.VIEW_tex_depth_odd] = self.depth_texture[1];
            bindings.views[shd_main.VIEW_tex_normal] = normal_view;
            bindings.samplers[shd_main.SMP_smp] = self.sampler;

            sg.applyPipeline(pip);
            sg.applyBindings(bindings);
            sg.applyUniforms(shd_main.UB_gtao_params, sg.asRange(&p));
            sg.draw(0, 3, 1);
        }

        self.frame +%= 1;

        // Passes 4 and 5: separable denoise, ao -> scratch -> ao, so the result
        // ends up back in `ao` and no third target is needed.
        const r: f32 = @floatFromInt(@min(self.settings.denoise_radius, 8));
        const inv_w = 1.0 / @as(f32, @floatFromInt(self.width));
        const inv_h = 1.0 / @as(f32, @floatFromInt(self.height));
        self.denoisePass(self.ao, self.scratch, .{ inv_w, 0 }, r);
        self.denoisePass(self.scratch, self.ao, .{ 0, inv_h }, r);
    }

    fn prefilterPass(self: *XeGtao, level: u32, source: sg.View, params: *PrefilterParams) void {
        const sig = PassSignature{
            .color_count = 1,
            .color_formats = blk: {
                var fmts: [4]sg.PixelFormat = @splat(.NONE);
                fmts[0] = .R32F;
                break :blk fmts;
            },
            .depth_format = .NONE,
            .sample_count = 1,
        };

        var action = sg.PassAction{};
        action.colors[0] = .{ .load_action = .DONTCARE };
        var att = sg.Attachments{};
        att.colors[0] = self.depth_attach[level];

        zupra.beginDrawingPass(.{ .action = action, .attachments = att });
        defer zupra.endDrawing();

        sg.applyViewport(0, 0, @intCast(self.mip_width[level]), @intCast(self.mip_height[level]), true);

        const pip = self.pipelineFor(self.prefilter_shader, sig) orelse return;
        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = self.vbuf;
        bindings.views[shd_pre.VIEW_tex_source] = source;
        bindings.samplers[shd_pre.SMP_smp] = self.sampler;

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(shd_pre.UB_prefilter_params, sg.asRange(params));
        sg.draw(0, 3, 1);
    }

    fn denoisePass(self: *XeGtao, src: Framebuffer, dst: Framebuffer, step: [2]f32, radius: f32) void {
        zupra.beginDrawingFramebuffer(dst);
        defer zupra.endDrawing();

        const pip = self.pipelineFor(self.denoise_shader, dst.passSignature()) orelse return;

        var p = DenoiseParams{
            .params = .{ step[0], step[1], radius, self.settings.denoise_tolerance },
        };

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = self.vbuf;
        bindings.views[shd_den.VIEW_tex_ao] = src.sample_view;
        // Level 0 is even, so it lives in image 0.
        bindings.views[shd_den.VIEW_tex_depth] = self.depth_texture[0];
        bindings.samplers[shd_den.SMP_smp] = self.sampler;

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(shd_den.UB_denoise_params, sg.asRange(&p));
        sg.draw(0, 3, 1);
    }

    fn pipelineFor(self: *XeGtao, shader: sg.Shader, sig: PassSignature) ?sg.Pipeline {
        return self.cache.get(.{
            .shader = shader,
            .layout = .fullscreen,
            .index_type = .u16,
            .indexed = false,
            .pass = sig,
            .primitive = .TRIANGLES,
            .cull = .NONE,
            .blend = .none,
            .depth_test = false,
            .depth_write = false,
        }) catch |err| {
            std.log.err("XeGtao: pipeline cache failed: {}", .{err});
            return null;
        };
    }

    /// The finished occlusion buffer, for the lighting pass to multiply in.
    pub fn aoView(self: XeGtao) sg.View {
        return self.ao.sample_view;
    }

    pub fn aoSampler(self: XeGtao) sg.Sampler {
        return self.sampler;
    }

    /// The AO buffer as a texture, for drawing to screen. Seeing the raw buffer
    /// is worth far more than inferring its contents from a lit image.
    pub fn debugTexture(self: XeGtao) Texture {
        return self.ao.asTexture();
    }
};

fn makeTarget(w: u32, h: u32) Framebuffer {
    return Framebuffer.init(.{
        .width = w,
        .height = h,
        // R8 is enough for a visibility scalar in [0,1]. The pyramid is where
        // precision matters, not here.
        .color_format = .R8,
        .depth_format = .NONE,
    });
}

fn scaled(width: u32, height: u32, s: Settings) struct { w: u32, h: u32 } {
    const div: u32 = if (s.half_resolution) 2 else 1;
    return .{ .w = @max(1, width / div), .h = @max(1, height / div) };
}
