//! src/render/taa.zig
//!
//! Temporal anti-aliasing.
//!
//! Renders each frame with the projection nudged by a fraction of a pixel, then
//! accumulates frames so that detail no single frame contains is recovered from
//! the sequence. That gives edge anti-aliasing far beyond what a post filter like
//! FXAA can reach -- and, just as usefully, averages out per-pixel noise in
//! stochastic effects. GTAO in particular is designed around this: every shipping
//! implementation runs a handful of slices and lets the temporal filter resolve
//! the rest, which is why TAA is a prerequisite for it rather than a companion.
//!
//! SCOPE: camera reprojection, not per-object motion vectors. Velocity is
//! derived from depth plus the previous view-projection, which is exact for
//! static geometry under any camera motion. An object that moved under its own
//! transform is not tracked, and relies on the resolve's neighbourhood clamp to
//! suppress ghosting.
//!
//! Adding true per-object velocity means storing each instance's PREVIOUS model
//! matrix and writing a velocity target in the geometry pass. That is a retained
//! -mode requirement: submission currently records a fresh list each frame and
//! discards it, so there is nowhere for last frame's transform to live. It is the
//! same architectural question the culling notes raise, reached from a different
//! direction.
//!
//! THE JITTER IS SUBTRACTED BACK OUT for reprojection. The matrices handed to the
//! resolve must be UNJITTERED, or every frame's offset would be baked into the
//! reprojection and the history would chase its own tail.

const std = @import("std");
const sg = @import("sokol").gfx;
const zupra = @import("../root.zig");
const math = @import("../math.zig");
const zm = math.zm;
const pipeline = @import("../graphics/pipeline.zig");
const gfx = @import("../graphics/graphics.zig");
const fb = @import("framebuffer.zig");
const Camera3D = @import("camera3d.zig").Camera3D;

const shd = @import("shaders").taa_resolve;

const PipelineCache = pipeline.PipelineCache;
const PipelineKey = pipeline.PipelineKey;
const Framebuffer = fb.Framebuffer;
const Texture = gfx.texture.Texture;
const Vertex2D = gfx.Vertex2D;
const Matrix = math.Matrix;

const TaaParams = extern struct {
    inv_view_proj: [16]f32,
    prev_view_proj: [16]f32,
    params: [4]f32, // 1/w, 1/h, blend weight, origin_top_left
    flags: [4]f32, // reset, variance gamma, history sharpening, unused
};

pub const Settings = struct {
    /// How much of each new frame is mixed in. Lower means a longer history:
    /// smoother edges and cleaner noise, but more ghosting behind anything the
    /// reprojection cannot follow. 0.1 is the usual starting point.
    blend: f32 = 0.1,
    /// Length of the jitter sequence. Powers of two land badly on Halton; 8 and
    /// 16 are the common choices, 16 resolving finer detail at the cost of
    /// taking longer to settle after a cut.
    sequence_length: u32 = 8,
    /// Scales the sub-pixel offset. Below 1 gives a tighter, sharper result that
    /// resolves less; above 1 softens. Rarely worth moving.
    jitter_scale: f32 = 1.0,
    /// How many standard deviations of the local neighbourhood the history may
    /// stray before it is clipped back.
    ///
    /// This is the ghosting/stability dial. Lower rejects history sooner, which
    /// kills trails behind moving objects but discards good history too and
    /// leaves more aliasing and noise. Higher accumulates longer and looks
    /// cleaner standing still, at the cost of ghosting in motion. 1.0 to 1.5 is
    /// the usual range.
    ///
    /// Note this is a TIGHTER test than a plain min/max neighbourhood box, which
    /// one outlier can stretch wide open. Coming from such a box, expect to want
    /// a value above 1.5 to accumulate as much history as before.
    variance_gamma: f32 = 1.75,
    /// How much of the sharpening bicubic to use when resampling the history,
    /// 0 = plain bilinear, 1 = full Catmull-Rom.
    ///
    /// The reprojected position rarely lands on a texel centre, so the history
    /// is resampled EVERY frame. Bilinear there is a low-pass filter applied
    /// hundreds of times over, which is why TAA is known for softening an image.
    /// A sharpening kernel cancels most of that.
    ///
    /// But it is a feedback loop -- the sharpened result becomes next frame's
    /// input -- so at full strength the overshoot compounds and draws bright rims
    /// along silhouettes against darker backgrounds. Partial strength keeps most
    /// of the sharpness with the accumulation staying below visibility. Raise it
    /// if the image feels soft; lower it if edges start to glow.
    history_sharpening: f32 = 0.4,
};

pub const Taa = struct {
    cache: *PipelineCache,
    shader: sg.Shader,
    sampler: sg.Sampler,
    vbuf: sg.Buffer,

    /// Two histories, ping-ponged: a resolve reads one and writes the other,
    /// because a target cannot be sampled while it is attached.
    history: [2]Framebuffer = undefined,
    /// Which history holds the PREVIOUS frame's result.
    read_index: u32 = 0,

    /// Last frame's unjittered view-projection, for reprojection.
    prev_view_proj: Matrix = undefined,
    /// Set whenever the history is meaningless: first frame, after a resize, or
    /// after the caller reports a cut. Without it, frame one blends against an
    /// uninitialised target.
    reset: bool = true,

    frame: u64 = 0,
    settings: Settings = .{},
    width: u32 = 0,
    height: u32 = 0,

    pub fn init(cache: *PipelineCache, width: u32, height: u32, settings: Settings) Taa {
        const top_left = sg.queryFeatures().origin_top_left;
        const clip = [3][2]f32{ .{ -1, -1 }, .{ 3, -1 }, .{ -1, 3 } };
        var verts: [3]Vertex2D = undefined;
        for (clip, 0..) |p, i| {
            const u = 0.5 + 0.5 * p[0];
            const v = if (top_left) 0.5 - 0.5 * p[1] else 0.5 + 0.5 * p[1];
            verts[i] = .{ .pos = p, .uv = .{ u, v }, .color = 0xFFFFFFFF };
        }

        var self = Taa{
            .cache = cache,
            .shader = sg.makeShader(shd.taaResolveShaderDesc(sg.queryBackend())),
            // LINEAR: the history is sampled at a reprojected position that
            // almost never lands on a texel centre, and point sampling there
            // would quantise the accumulation and undo the sub-pixel detail the
            // jitter exists to gather.
            .sampler = sg.makeSampler(.{
                .min_filter = .LINEAR,
                .mag_filter = .LINEAR,
                .wrap_u = .CLAMP_TO_EDGE,
                .wrap_v = .CLAMP_TO_EDGE,
            }),
            .vbuf = sg.makeBuffer(.{ .data = sg.asRange(&verts) }),
            .settings = settings,
            .prev_view_proj = zm.identity(),
        };
        self.build(width, height);
        return self;
    }

    pub fn deinit(self: *Taa) void {
        for (&self.history) |*h| h.deinit();
        sg.destroyBuffer(self.vbuf);
        sg.destroySampler(self.sampler);
        sg.destroyShader(self.shader);
    }

    pub fn resize(self: *Taa, width: u32, height: u32) void {
        if (width == self.width and height == self.height) return;
        for (&self.history) |*h| h.deinit();
        self.build(width, height);
    }

    fn build(self: *Taa, width: u32, height: u32) void {
        for (&self.history) |*h| {
            h.* = Framebuffer.init(.{
                .width = width,
                .height = height,
                // RGBA16F: this holds pre-tonemap light, and clamping the
                // history to 1.0 would quietly crush every highlight the moment
                // it entered the accumulation.
                .color_format = .RGBA16F,
            });
        }
        self.width = width;
        self.height = height;
        // New targets hold nothing worth blending against.
        self.reset = true;
    }

    /// Discard the accumulated history. Call on a camera cut, a teleport, or any
    /// discontinuity: reprojection assumes the previous frame showed roughly the
    /// same scene, and blending across a cut smears one shot into the next.
    pub fn resetHistory(self: *Taa) void {
        self.reset = true;
    }

    /// Sub-pixel offset for this frame, in NDC. Add to the projection before
    /// rendering; see the note in the wiring doc.
    ///
    /// Halton (2,3), which fills the pixel far more evenly than a random
    /// sequence of the same length -- clumped samples would leave parts of the
    /// pixel unsampled and the accumulation would converge to the wrong answer.
    pub fn jitter(self: Taa) [2]f32 {
        const i: u32 = @intCast(self.frame % @max(1, self.settings.sequence_length));
        // Centred on zero: halton is in [0,1), so shift to [-0.5, 0.5).
        const jx = (halton(i + 1, 2) - 0.5) * self.settings.jitter_scale;
        const jy = (halton(i + 1, 3) - 0.5) * self.settings.jitter_scale;
        // Pixels -> NDC. NDC spans 2 across the target, hence the factor.
        return .{
            jx * 2.0 / @as(f32, @floatFromInt(@max(1, self.width))),
            jy * 2.0 / @as(f32, @floatFromInt(@max(1, self.height))),
        };
    }

    /// Resolve `color` against the history and return the accumulated result.
    ///
    /// `view_proj` must be UNJITTERED -- the jitter is a rendering offset, not
    /// part of where the surface actually is, and feeding it in here would make
    /// the reprojection chase the jitter sequence instead of the camera.
    pub fn resolve(self: *Taa, color: Texture, depth_view: sg.View, view_proj: Matrix) Texture {
        const write_index = 1 - self.read_index;
        const dst = self.history[write_index];

        const pip = self.cache.get(.{
            .shader = self.shader,
            .layout = .fullscreen,
            .index_type = .u16,
            .indexed = false,
            .pass = dst.passSignature(),
            .primitive = .TRIANGLES,
            .cull = .NONE,
            .blend = .none,
            .depth_test = false,
            .depth_write = false,
        }) catch |err| {
            std.log.err("Taa: pipeline cache failed: {}", .{err});
            return color;
        };

        var params = TaaParams{
            .inv_view_proj = @bitCast(zm.inverse(view_proj)),
            .prev_view_proj = @bitCast(self.prev_view_proj),
            .params = .{
                1.0 / @as(f32, @floatFromInt(self.width)),
                1.0 / @as(f32, @floatFromInt(self.height)),
                std.math.clamp(self.settings.blend, 0.01, 1.0),
                if (sg.queryFeatures().origin_top_left) 1.0 else 0.0,
            },
            .flags = .{
                if (self.reset) 1.0 else 0.0,
                self.settings.variance_gamma,
                self.settings.history_sharpening,
                0,
            },
        };

        zupra.beginDrawingFramebufferClear(dst, .{ .r = 0, .g = 0, .b = 0, .a = 1 });
        {
            var bindings = sg.Bindings{};
            bindings.vertex_buffers[0] = self.vbuf;
            bindings.views[shd.VIEW_tex_color] = color.view;
            bindings.views[shd.VIEW_tex_history] = self.history[self.read_index].sample_view;
            bindings.views[shd.VIEW_tex_depth] = depth_view;
            bindings.samplers[shd.SMP_smp] = self.sampler;

            sg.applyPipeline(pip);
            sg.applyBindings(bindings);
            sg.applyUniforms(shd.UB_taa_params, sg.asRange(&params));
            sg.draw(0, 3, 1);
        }
        zupra.endDrawing();

        // Advance. The frame counter drives the jitter sequence, so it must tick
        // exactly once per rendered frame.
        self.read_index = write_index;
        self.prev_view_proj = view_proj;
        self.reset = false;
        self.frame +%= 1;

        return dst.asTexture();
    }
};

/// Radical inverse of `index` in `base`. Halton fills the unit interval evenly
/// at every prefix length, which matters here: the accumulation is a running
/// average over however many frames have elapsed, so a sequence that is only
/// well distributed once complete would bias every partial result.
fn halton(index: u32, base: u32) f32 {
    var f: f32 = 1.0;
    var r: f32 = 0.0;
    var i = index;
    const b: f32 = @floatFromInt(base);
    while (i > 0) {
        f /= b;
        r += f * @as(f32, @floatFromInt(i % base));
        i /= base;
    }
    return r;
}
