//! src/render/bloom.zig
//!
//! HDR bloom via a progressive downsample / upsample mip chain.
//!
//! Emissive surfaces already write HDR values, but without bloom a value of 40
//! and a value of 4 both tonemap to roughly white — so an emissive material
//! reads as a bright patch rather than as something emitting light. Bloom is
//! what makes the difference visible: the brighter a surface is, the further its
//! light spreads into its surroundings.
//!
//! WHY A MIP CHAIN rather than one big blur. Real lens bloom has no single
//! radius. It is a tight bright core inside a wide faint halo, and reproducing
//! that with a single Gaussian means an enormous kernel that is both expensive
//! and still wrong — one radius cannot be two. Halving the image repeatedly and
//! summing on the way back up gives every scale at once, and the widest
//! contributions are computed on the smallest images, so most of the visual
//! reach costs almost nothing.
//!
//! RUNS BEFORE TONEMAP. Bloom is optical: light scatters on its way to the
//! sensor, while values are still physical. Applying it after tonemap would mean
//! doing display-referred arithmetic to something that happened in linear light,
//! and bright highlights would clip rather than bloom.
//!
//! Not a PostEffect: that contract is one input texture to one output at a fixed
//! resolution, and this needs a pyramid. Same reason anti-aliasing is a built-in
//! rather than an effect.

const std = @import("std");
const sg = @import("sokol").gfx;
const zupra = @import("../root.zig");
const pipeline = @import("../graphics/pipeline.zig");
const gfx = @import("../graphics/graphics.zig");
const fb = @import("framebuffer.zig");

const shd_pre = @import("shaders").bloom_prefilter;
const shd_down = @import("shaders").bloom_downsample;
const shd_up = @import("shaders").bloom_upsample;
const shd_comp = @import("shaders").bloom_composite;

const PipelineCache = pipeline.PipelineCache;
const PipelineKey = pipeline.PipelineKey;
const PassSignature = pipeline.PassSignature;
const Framebuffer = fb.Framebuffer;
const Texture = gfx.texture.Texture;
const Vertex2D = gfx.Vertex2D;

/// Hard cap on chain length. Seven levels from 1080p reaches an 8-pixel-tall
/// image, which is already wider than any bloom anyone wants; beyond that the
/// passes cost more in fixed overhead than they contribute.
pub const max_mips = 7;

pub const Settings = struct {
    /// Luminance above which a pixel starts to bloom, in linear light. Values
    /// near 1.0 mean "anything brighter than white paper", which is usually what
    /// you want — the sun, emissive panels, specular glints.
    threshold: f32 = 1.0,
    /// Width of the soft ramp around the threshold. Zero gives a hard cut, and a
    /// surface dimming through it will visibly pop rather than fade.
    knee: f32 = 0.5,
    /// How much bloom is added back. This is the main artistic dial; above ~0.2
    /// the image starts to look hazy rather than bright.
    intensity: f32 = 0.06,
    /// Tent-filter radius during upsampling, in source texels. Larger spreads
    /// the halo further at the cost of softening the core.
    filter_radius: f32 = 1.0,
    /// Chain length. Fewer levels means a tighter, more local glow; more means a
    /// wider atmospheric one. Clamped to what the resolution can actually
    /// support and to max_mips.
    mip_count: u32 = 6,
};

pub const Bloom = struct {
    cache: *PipelineCache,
    prefilter_shader: sg.Shader,
    down_shader: sg.Shader,
    up_shader: sg.Shader,
    composite_shader: sg.Shader,
    sampler: sg.Sampler,
    vbuf: sg.Buffer,

    /// The pyramid. mips[0] is half the render size, each subsequent one half
    /// again. Bilinear sampling of the level below is what makes the 13-tap
    /// downsample cover as much ground as it does, so LINEAR is required here,
    /// not merely preferred.
    mips: [max_mips]Framebuffer = undefined,
    mip_count: u32 = 0,

    /// Scene + bloom. A separate target because the composite reads the scene
    /// colour and cannot also write to it.
    output: Framebuffer = undefined,

    settings: Settings = .{},
    width: u32 = 0,
    height: u32 = 0,

    pub fn init(cache: *PipelineCache, width: u32, height: u32, settings: Settings) Bloom {
        // Fullscreen triangle with backend-correct UV origin, same construction
        // PostChain uses. Duplicated rather than shared because coupling bloom's
        // lifetime to the post chain's buys nothing.
        const top_left = sg.queryFeatures().origin_top_left;
        const clip = [3][2]f32{ .{ -1, -1 }, .{ 3, -1 }, .{ -1, 3 } };
        var verts: [3]Vertex2D = undefined;
        for (clip, 0..) |p, i| {
            const u = 0.5 + 0.5 * p[0];
            const v = if (top_left) 0.5 - 0.5 * p[1] else 0.5 + 0.5 * p[1];
            verts[i] = .{ .pos = p, .uv = .{ u, v }, .color = 0xFFFFFFFF };
        }

        var self = Bloom{
            .cache = cache,
            .prefilter_shader = sg.makeShader(shd_pre.bloomPrefilterShaderDesc(sg.queryBackend())),
            .down_shader = sg.makeShader(shd_down.bloomDownsampleShaderDesc(sg.queryBackend())),
            .up_shader = sg.makeShader(shd_up.bloomUpsampleShaderDesc(sg.queryBackend())),
            .composite_shader = sg.makeShader(shd_comp.bloomCompositeShaderDesc(sg.queryBackend())),
            .sampler = sg.makeSampler(.{
                .min_filter = .LINEAR,
                .mag_filter = .LINEAR,
                .wrap_u = .CLAMP_TO_EDGE,
                .wrap_v = .CLAMP_TO_EDGE,
            }),
            .vbuf = sg.makeBuffer(.{ .data = sg.asRange(&verts) }),
            .settings = settings,
        };
        self.build(width, height);
        return self;
    }

    pub fn deinit(self: *Bloom) void {
        self.destroyTargets();
        sg.destroyBuffer(self.vbuf);
        sg.destroySampler(self.sampler);
        sg.destroyShader(self.composite_shader);
        sg.destroyShader(self.up_shader);
        sg.destroyShader(self.down_shader);
        sg.destroyShader(self.prefilter_shader);
    }

    pub fn resize(self: *Bloom, width: u32, height: u32) void {
        if (width == self.width and height == self.height) return;
        self.destroyTargets();
        self.build(width, height);
    }

    /// Rebuild after a settings change that affects the chain (mip_count).
    pub fn rebuild(self: *Bloom) void {
        const w = self.width;
        const h = self.height;
        self.destroyTargets();
        self.build(w, h);
    }

    fn destroyTargets(self: *Bloom) void {
        var i: u32 = 0;
        while (i < self.mip_count) : (i += 1) self.mips[i].deinit();
        if (self.width != 0) self.output.deinit();
        self.mip_count = 0;
    }

    fn build(self: *Bloom, width: u32, height: u32) void {
        self.width = width;
        self.height = height;
        self.output = Framebuffer.init(.{
            .width = width,
            .height = height,
            .color_format = .RGBA16F,
        });

        var w = width / 2;
        var h = height / 2;
        var count: u32 = 0;
        const wanted = @min(self.settings.mip_count, max_mips);
        while (count < wanted and w >= 2 and h >= 2) : (count += 1) {
            self.mips[count] = Framebuffer.init(.{
                .width = w,
                .height = h,
                // RGBA16F throughout: these hold pre-tonemap light, and a bright
                // emissive can easily be 40.0. RGBA8 would clip it to 1.0 in the
                // very first pass and there would be nothing left to bloom.
                .color_format = .RGBA16F,
            });
            w /= 2;
            h /= 2;
        }
        self.mip_count = count;
    }

    /// Produce scene + bloom. Returns the scene texture untouched when the chain
    /// is unusable (no mips at a tiny window), so the caller never has to branch.
    pub fn render(self: *Bloom, scene: Texture) Texture {
        if (self.mip_count == 0) return scene;

        // 1) Bright-pass into the first mip, halving resolution as it goes.
        {
            var p = shd_pre.PrefilterParams{ .params = .{
                self.settings.threshold,
                self.settings.knee,
                1.0 / @as(f32, @floatFromInt(self.width)),
                1.0 / @as(f32, @floatFromInt(self.height)),
            } };
            self.pass(self.mips[0], self.prefilter_shader, scene.view, null, shd_pre.UB_prefilter_params, std.mem.asBytes(&p), false);
        }

        // 2) Down the chain. Each level is a resample of the one above, so the
        // texel size passed is the SOURCE's, not the destination's.
        var i: u32 = 1;
        while (i < self.mip_count) : (i += 1) {
            const src = self.mips[i - 1];
            var p = shd_down.DownsampleParams{ .params = .{
                1.0 / @as(f32, @floatFromInt(src.width)),
                1.0 / @as(f32, @floatFromInt(src.height)),
                0,
                0,
            } };
            self.pass(self.mips[i], self.down_shader, src.sample_view, null, shd_down.UB_downsample_params, std.mem.asBytes(&p), false);
        }

        // 3) Back up, adding each level onto the one above it. Additive blending
        // is what makes this a sum over scales rather than a sequence of
        // overwrites -- every mip's contribution survives to the top.
        i = self.mip_count - 1;
        while (i > 0) : (i -= 1) {
            const src = self.mips[i];
            const radius = self.settings.filter_radius;
            var p = shd_up.UpsampleParams{ .params = .{
                radius / @as(f32, @floatFromInt(src.width)),
                radius / @as(f32, @floatFromInt(src.height)),
                1.0,
                0,
            } };
            self.pass(self.mips[i - 1], self.up_shader, src.sample_view, null, shd_up.UB_upsample_params, std.mem.asBytes(&p), true);
        }

        // 4) Scene + bloom, still in linear light.
        {
            var p = shd_comp.CompositeParams{ .params = .{ self.settings.intensity, 0, 0, 0 } };
            self.pass(self.output, self.composite_shader, scene.view, self.mips[0].sample_view, shd_comp.UB_composite_params, std.mem.asBytes(&p), false);
        }

        return self.output.asTexture();
    }

    /// One fullscreen pass. `additive` selects blend-onto-existing rather than
    /// replace, and also decides whether the target is cleared: an additive pass
    /// must keep what is already in the destination.
    fn pass(
        self: *Bloom,
        target: Framebuffer,
        shader: sg.Shader,
        view0: sg.View,
        view1: ?sg.View,
        ub_slot: u32,
        uniforms: []const u8,
        additive: bool,
    ) void {
        const key = PipelineKey{
            .shader = shader,
            .layout = .fullscreen,
            .index_type = .u16,
            .indexed = false,
            .pass = target.passSignature(),
            .primitive = .TRIANGLES,
            .cull = .NONE,
            .blend = if (additive) .additive else .none,
            .depth_test = false,
            .depth_write = false,
        };
        const pip = self.cache.get(key) catch |err| {
            std.log.err("Bloom: pipeline cache failed: {}", .{err});
            return;
        };

        if (additive) {
            zupra.beginDrawingFramebufferLoad(target);
        } else {
            zupra.beginDrawingFramebufferClear(target, .{ .r = 0, .g = 0, .b = 0, .a = 1 });
        }
        defer zupra.endDrawing();

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = self.vbuf;
        bindings.views[0] = view0;
        if (view1) |v| bindings.views[1] = v;
        bindings.samplers[0] = self.sampler;

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(ub_slot, sg.asRange(uniforms));
        sg.draw(0, 3, 1);
    }
};
