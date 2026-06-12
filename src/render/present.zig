//! src/render/present.zig
//!
//! Present pass: tonemaps the linear-HDR scene-color target onto the swapchain.
//! The final step both forward and deferred renderers end with. Owns the
//! tonemap shader + a fullscreen triangle; `exposure` is adjustable.

const std = @import("std");
const sg = @import("sokol").gfx;

const gfx = @import("../graphics/graphics.zig");
const pipeline = @import("../graphics/pipeline.zig");
const tex = @import("../graphics/texture.zig");
const FullscreenTriangle = @import("fullscreen.zig").FullscreenTriangle;

const shd = @import("shaders").tonemap;

const Texture = tex.Texture;
const PipelineCache = pipeline.PipelineCache;
const PipelineKey = pipeline.PipelineKey;
const PassSignature = pipeline.PassSignature;

const TonemapParams = extern struct { params: [4]f32 };

pub const Present = struct {
    cache: *PipelineCache,
    shader: sg.Shader,
    sampler: sg.Sampler,
    tri: FullscreenTriangle,
    exposure: f32 = 1.0,

    pub fn init(cache: *PipelineCache) Present {
        return .{
            .cache = cache,
            .shader = sg.makeShader(shd.tonemapShaderDesc(sg.queryBackend())),
            .sampler = sg.makeSampler(.{
                .min_filter = .LINEAR,
                .mag_filter = .LINEAR,
                .wrap_u = .CLAMP_TO_EDGE,
                .wrap_v = .CLAMP_TO_EDGE,
            }),
            .tri = FullscreenTriangle.init(),
        };
    }

    pub fn deinit(self: *Present) void {
        self.tri.deinit();
        sg.destroySampler(self.sampler);
        sg.destroyShader(self.shader);
    }

    /// Draw `hdr` (a linear-HDR scene-color texture) tonemapped into the current
    /// pass (typically the swapchain). Call inside a begun pass.
    pub fn render(self: *Present, hdr: Texture, pass: PassSignature) void {
        const key = PipelineKey{
            .shader = self.shader,
            .layout = .fullscreen, // fullscreen tri: pos + uv (color slot unused)
            .index_type = .u32,
            .indexed = false,
            .pass = pass,
            .primitive = .TRIANGLES,
            .cull = .NONE,
            .blend = .none,
            .depth_test = false,
            .depth_write = false,
        };
        const pip = self.cache.get(key) catch |err| {
            std.log.err("Present: pipeline cache failed: {}", .{err});
            return;
        };

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = self.tri.vbuf;
        bindings.views[shd.VIEW_tex_hdr] = hdr.view;
        bindings.samplers[shd.SMP_smp] = self.sampler;

        var params = TonemapParams{ .params = .{ self.exposure, 0, 0, 0 } };

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(shd.UB_tonemap_params, sg.asRange(&params));
        sg.draw(0, 3, 1);
    }
};
