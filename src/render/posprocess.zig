//! src/render/postprocess.zig
//!
//! Post-process chain: takes the scene's HDR scene-color and produces the final
//! swapchain image, running a sequence of fullscreen passes. Anti-aliasing is a
//! SWAPPABLE stage here (method chosen at runtime), which is how engines do it —
//! FXAA/SMAA/TAA are peers behind one interface; MSAA is deliberately NOT here
//! (it's a property of the scene-color target, a different axis — see notes).
//!
//! Chain order (AA sits after tonemap because FXAA/SMAA operate on perceptual/
//! LDR color; TAA would instead run before tonemap on HDR + need motion vectors):
//!
//!     scene_color(HDR) --tonemap--> ldr --AA--> swapchain
//!
//! When aa == .none, tonemap goes straight to the swapchain (no intermediate),
//! matching the original present path exactly.

const std = @import("std");
const sg = @import("sokol").gfx;
const zupra = @import("../root.zig");
const pipeline = @import("../graphics/pipeline.zig");
const gfx = @import("../graphics/graphics.zig");
const fb = @import("framebuffer.zig");
const present_mod = @import("present.zig");
const shd_fxaa = @import("shaders").fxaa;
const shd_fxaaq = @import("shaders").fxaa_quality;

const PipelineCache = pipeline.PipelineCache;
const PipelineKey = pipeline.PipelineKey;
const PassSignature = pipeline.PassSignature;
const Framebuffer = fb.Framebuffer;
const Present = present_mod.Present;
const Vertex2D = gfx.Vertex2D;
const Color = zupra.Color;
const Texture = gfx.texture.Texture;

/// Image-space AA methods (peers). MSAA is intentionally absent because it's hardware
/// multisampling on the render target, configured on the scene-color framebuffer,
/// not a post pass. Selecting an unimplemented method falls back to .none.
pub const AAMethod = enum {
    none,
    fxaa, // cheap console variant: 9 taps, no edge search
    fxaa_quality, // FXAA 3.11: searches the edge span, keeps texture detail
    // smaa,       // needs precomputed area/search LUTs
    // taa,        // needs velocity buffer + history + camera jitter

};

pub const PostChain = struct {
    cache: *PipelineCache,
    present_: Present, // owns the tonemap pass
    aa: AAMethod = .none,

    // FXAA resources
    fxaa_shader: sg.Shader,
    fxaaq_shader: sg.Shader,
    sampler: sg.Sampler,
    fullscreen_vbuf: sg.Buffer,

    /// Min local luma contrast to treat as an edge. Lower = more edges
    /// caught (softer), higher = only strong edges (sharper, more aliasing).
    edge_threshold: f32 = 0.166,

    /// How much sub-pixel detail gets low-passed. 0 = none (crisp but
    /// shimmery on thin geometry), 1 = max (smooth but softer).
    subpix_quality: f32 = 0.75,

    // LDR intermediate (tonemap output when AA is active)
    ldr: Framebuffer,
    width: u32,
    height: u32,
    clear_color: Color,

    pub fn init(cache: *PipelineCache, width: u32, height: u32, clear_color: Color) PostChain {
        const fxaa_shader = sg.makeShader(shd_fxaa.fxaaShaderDesc(sg.queryBackend()));
        const fxaaq_shader = sg.makeShader(shd_fxaaq.fxaaQualityShaderDesc(sg.queryBackend()));
        const sampler = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        // Fullscreen triangle, backend-correct UV origin (same convention as the
        // deferred lighting pass / present).
        const top_left = sg.queryFeatures().origin_top_left;
        const clip = [3][2]f32{ .{ -1, -1 }, .{ 3, -1 }, .{ -1, 3 } };
        var verts: [3]Vertex2D = undefined;
        for (clip, 0..) |p, i| {
            const u = 0.5 + 0.5 * p[0];
            const v = if (top_left) 0.5 - 0.5 * p[1] else 0.5 + 0.5 * p[1];
            verts[i] = .{ .pos = p, .uv = .{ u, v }, .color = 0xFFFFFFFF };
        }
        const vbuf = sg.makeBuffer(.{ .data = sg.asRange(&verts) });

        return PostChain{
            .cache = cache,
            .present_ = Present.init(cache),
            .fxaa_shader = fxaa_shader,
            .fxaaq_shader = fxaaq_shader,
            .sampler = sampler,
            .fullscreen_vbuf = vbuf,
            .ldr = Framebuffer.init(.{ .width = width, .height = height, .color_format = .RGBA8 }), // LDR tonemap target
            .width = width,
            .height = height,
            .clear_color = clear_color,
        };
    }

    pub fn deinit(self: *PostChain) void {
        self.ldr.deinit();
        self.present_.deinit();
        sg.destroyBuffer(self.fullscreen_vbuf);
        sg.destroySampler(self.sampler);
        sg.destroyShader(self.fxaa_shader);
        sg.destroyShader(self.fxaaq_shader);
    }

    pub fn resize(self: *PostChain, width: u32, height: u32) void {
        if (width == self.width and height == self.height) return;
        self.ldr.deinit();
        self.ldr = Framebuffer.init(.{ .width = width, .height = height, .color_format = .RGBA8 });
        self.width = width;
        self.height = height;
    }

    /// Produce the final swapchain image from the HDR scene color.
    pub fn present(self: *PostChain, scene_color: Texture) void {
        const effective: AAMethod = self.aa;

        switch (effective) {
            .none => {
                // Tonemap straight to the swapchain (original path).
                zupra.beginDrawingClear(self.clear_color);
                self.present_.render(scene_color, PassSignature.swapchainPass());
                zupra.endDrawing();
            },
            .fxaa, .fxaa_quality => {
                // 1) Tonemap HDR scene-color into the LDR intermediate.
                zupra.beginDrawingFramebufferClear(self.ldr, self.clear_color);
                self.present_.render(scene_color, self.ldr.passSignature());
                zupra.endDrawing();

                // 2) FXAA the LDR image to the swapchain.
                zupra.beginDrawingClear(self.clear_color);
                if (effective == .fxaa) {
                    self.fxaa(self.ldr.asTexture(), PassSignature.swapchainPass());
                } else {
                    self.fxaaQuality(self.ldr.asTexture(), PassSignature.swapchainPass());
                }
                zupra.endDrawing();
            },
        }
    }

    fn fxaa(self: *PostChain, input: Texture, pass: PassSignature) void {
        const key = PipelineKey{
            .shader = self.fxaa_shader,
            .layout = .fullscreen,
            .index_type = .u16, //unused
            .indexed = false,
            .pass = pass,
            .primitive = .TRIANGLES,
            .cull = .NONE,
            .blend = .none,
            .depth_test = false,
            .depth_write = false,
        };
        const pip = self.cache.get(key) catch return;

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = self.fullscreen_vbuf;
        bindings.views[shd_fxaa.VIEW_scene_ldr] = input.view;
        bindings.samplers[shd_fxaa.SMP_smp] = self.sampler;

        var fs = shd_fxaa.FsParams{ .inv_resolution = .{
            1.0 / @as(f32, @floatFromInt(self.width)),
            1.0 / @as(f32, @floatFromInt(self.height)),
            0,
            0,
        } };

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(shd_fxaa.UB_fs_params, sg.asRange(&fs));
        sg.draw(0, 3, 1);
    }

    fn fxaaQuality(self: *PostChain, input: Texture, pass: PassSignature) void {
        const key = PipelineKey{
            .shader = self.fxaaq_shader,
            .layout = .fullscreen,
            .index_type = .u16, // unused
            .indexed = false,
            .pass = pass,
            .primitive = .TRIANGLES,
            .cull = .NONE,
            .blend = .none,
            .depth_test = false,
            .depth_write = false,
        };
        const pip = self.cache.get(key) catch return;

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = self.fullscreen_vbuf;
        bindings.views[shd_fxaaq.VIEW_scene_ldr] = input.view;
        bindings.samplers[shd_fxaaq.SMP_smp] = self.sampler;

        var fs = shd_fxaaq.FsParams{ .inv_resolution = .{
            1.0 / @as(f32, @floatFromInt(self.width)),
            1.0 / @as(f32, @floatFromInt(self.height)),
            self.edge_threshold,
            self.subpix_quality,
        } };

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(shd_fxaaq.UB_fs_params, sg.asRange(&fs));
        sg.draw(0, 3, 1);
    }
};
