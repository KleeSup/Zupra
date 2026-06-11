//! src/render/gbuffer.zig
//!
//! Deferred-rendering G-buffer: a multi-target offscreen render target that the
//! geometry pass writes *surface data* into (not lit color), which the lighting
//! pass then samples to compute PBR shading once per screen pixel.
//!
//! Layout (4 color targets + depth), single-sampled (deferred G-buffers don't do
//! MSAA because averaging raw normals/material data is meaningless. Rather AA is done later
//! in a post pass like TAA, FXAA or SMAA):
//!   RT0 albedo    RGBA8    base color (rgb) + (a free)
//!   RT1 normal    RGBA16F  world-space normal (xyz)
//!   RT2 position  RGBA16F  world-space position (xyz)  [stored, not yet
//!                          reconstructed from depth -> simpler & correct first;
//!                          depth-reconstruction is a bandwidth optimization]
//!   RT3 material  RGBA8    metallic (r) / roughness (g) / ao (b)
//!   depth         DEPTH    geometry depth test
//!

const std = @import("std");
const sg = @import("sokol").gfx;
const pipeline = @import("../graphics/pipeline.zig");
const tex = @import("../graphics/texture.zig");
const zupra = @import("../root.zig");

const PassSignature = pipeline.PassSignature;
const Texture = tex.Texture;
const Color = zupra.Color;

pub const gbuffer_color_count = 4;
const albedo_format: sg.PixelFormat = .RGBA8;
const normal_format: sg.PixelFormat = .RGBA16F;
const position_format: sg.PixelFormat = .RGBA16F;
const material_format: sg.PixelFormat = .RGBA8;
const depth_format: sg.PixelFormat = .DEPTH;

/// One color render target: the image, its attachment view (render into), and a
/// texture view (sample afterward).
const Target = struct {
    img: sg.Image,
    attach: sg.View,
    sample: sg.View,
    format: sg.PixelFormat,

    fn init(w: i32, h: i32, format: sg.PixelFormat) Target {
        const img = sg.makeImage(.{
            .width = w,
            .height = h,
            .pixel_format = format,
            .sample_count = 1,
            .usage = .{ .color_attachment = true },
        });
        return .{
            .img = img,
            .attach = sg.makeView(.{ .color_attachment = .{ .image = img } }),
            .sample = sg.makeView(.{ .texture = .{ .image = img } }),
            .format = format,
        };
    }

    fn deinit(self: Target) void {
        sg.destroyView(self.sample);
        sg.destroyView(self.attach);
        sg.destroyImage(self.img);
    }

    fn texture(self: Target, w: u32, h: u32) Texture {
        return .{ .img = self.img, .view = self.sample, .width = w, .height = h, .format = self.format };
    }
};

pub const GBuffer = struct {
    width: u32,
    height: u32,

    albedo: Target,
    normal: Target,
    position: Target,
    material: Target,

    depth_img: sg.Image,
    depth_view: sg.View,

    pub fn init(width: u32, height: u32) GBuffer {
        const w: i32 = @intCast(width);
        const h: i32 = @intCast(height);

        const depth_img = sg.makeImage(.{
            .width = w,
            .height = h,
            .pixel_format = depth_format,
            .sample_count = 1,
            .usage = .{ .depth_stencil_attachment = true },
        });

        return .{
            .width = width,
            .height = height,
            .albedo = Target.init(w, h, albedo_format),
            .normal = Target.init(w, h, normal_format),
            .position = Target.init(w, h, position_format),
            .material = Target.init(w, h, material_format),
            .depth_img = depth_img,
            .depth_view = sg.makeView(.{ .depth_stencil_attachment = .{ .image = depth_img } }),
        };
    }

    pub fn deinit(self: *GBuffer) void {
        self.albedo.deinit();
        self.normal.deinit();
        self.position.deinit();
        self.material.deinit();
        sg.destroyView(self.depth_view);
        sg.destroyImage(self.depth_img);
    }

    /// Signature the geometry renderer passes to begin() so its pipelines target
    /// these 4 attachments.
    pub fn passSignature(self: GBuffer) PassSignature {
        _ = self;
        var sig = PassSignature{
            .color_count = gbuffer_color_count,
            .depth_format = depth_format,
            .sample_count = 1,
        };
        sig.color_formats[0] = albedo_format;
        sig.color_formats[1] = normal_format;
        sig.color_formats[2] = position_format;
        sig.color_formats[3] = material_format;
        return sig;
    }

    /// sokol Pass writing into all 4 targets + depth. Clears all to zero so
    /// background pixels carry no surface data (albedo 0 -> no light).
    pub fn pass(self: GBuffer) sg.Pass {
        var action = sg.PassAction{};
        const zero = Color{ .r = 0, .g = 0, .b = 0, .a = 0 };
        action.colors[0] = .{ .load_action = .CLEAR, .clear_value = zero };
        action.colors[1] = .{ .load_action = .CLEAR, .clear_value = zero };
        action.colors[2] = .{ .load_action = .CLEAR, .clear_value = zero };
        action.colors[3] = .{ .load_action = .CLEAR, .clear_value = zero };
        action.depth = .{ .load_action = .CLEAR, .clear_value = 1.0 };

        var att = sg.Attachments{};
        att.colors[0] = self.albedo.attach;
        att.colors[1] = self.normal.attach;
        att.colors[2] = self.position.attach;
        att.colors[3] = self.material.attach;
        att.depth_stencil = self.depth_view;

        return .{ .action = action, .attachments = att };
    }

    // sampling textures for the lighting pass
    pub fn albedoTexture(self: GBuffer) Texture {
        return self.albedo.texture(self.width, self.height);
    }
    pub fn normalTexture(self: GBuffer) Texture {
        return self.normal.texture(self.width, self.height);
    }
    pub fn positionTexture(self: GBuffer) Texture {
        return self.position.texture(self.width, self.height);
    }
    pub fn materialTexture(self: GBuffer) Texture {
        return self.material.texture(self.width, self.height);
    }
};
