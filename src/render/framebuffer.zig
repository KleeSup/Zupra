//! src/render/framebuffer.zig
//!
//! Offscreen render target (FBO). Render scene content into this instead of the
//! swapchain, then sample its color texture in a later pass (post-processing,
//! deferred lighting, IBL prefiltering, UI-to-texture, etc).
//!
//! It owns:
//!   * a color image + its color-attachment view (to render into) + a texture
//!     view (to sample afterward),
//!   * optionally a depth image + depth-attachment view (required for 3D),
//!   * optionally, when sample_count is greater than one (MSAA): the color/depth images
//!     are multisampled and a separate single-sampled `resolve` image is the
//!     thing sampled.
//!

const std = @import("std");
const sg = @import("sokol").gfx;
const zupra = @import("../root.zig");
const pipeline = @import("../graphics/pipeline.zig");
const tex = @import("../graphics/texture.zig");

const PassSignature = pipeline.PassSignature;
const Texture = tex.Texture;
const Color = zupra.Color;

pub const FramebufferOptions = struct {
    width: u32,
    height: u32,
    color_format: sg.PixelFormat = .RGBA8,
    /// .NONE for a color-only target (no depth test in the pass).
    depth_format: sg.PixelFormat = .DEPTH,
    /// 1 = no MSAA. >1 allocates multisampled attachments + a resolve image.
    sample_count: u8 = 1,
};

pub const PassOptions = struct {
    action: sg.PassAction,
    /// Override the depth-stencil attachment. Deferred forward sub-passes attach
    /// the G-BUFFER's depth (so they depth-test/write against deferred geometry)
    /// rather than this framebuffer's own. null = use this framebuffer's depth.
    depth_view: ?sg.View = null,
    /// Whether this pass resolves MSAA into the sampleable image at endPass.
    /// Only meaningful when sample_count > 1. Leave true unless you're chaining
    /// several passes into the same target and only the LAST one needs to
    /// resolve — an MSAA resolve of an RGBA16F target isn't free, so skipping
    /// the intermediate ones is a real saving.
    resolve: bool = true,
};

pub const Framebuffer = struct {
    width: u32,
    height: u32,
    color_format: sg.PixelFormat,
    depth_format: sg.PixelFormat,
    sample_count: u8,

    // rendering targets
    color_img: sg.Image,
    color_view: sg.View, // color attachment (render into)
    depth_img: sg.Image = .{}, // .id == 0 if no depth
    depth_view: sg.View = .{},

    // MSAA resolve (sample_count == 1 -> resolve_img is the color_img itself)
    resolve_img: sg.Image = .{},
    resolve_view: sg.View = .{}, // resolve attachment for sokol

    // what you sample afterward (resolve image if MSAA, else the color image)
    sample_view: sg.View,

    pub fn init(opts: FramebufferOptions) Framebuffer {
        const msaa = opts.sample_count > 1;
        const w: i32 = @intCast(opts.width);
        const h: i32 = @intCast(opts.height);

        // Color (possibly multisampled) render image + its attachment view.
        const color_img = sg.makeImage(.{
            .width = w,
            .height = h,
            .pixel_format = opts.color_format,
            .sample_count = @intCast(opts.sample_count),
            .usage = .{ .color_attachment = true },
        });
        const color_view = sg.makeView(.{ .color_attachment = .{ .image = color_img } });

        var fb = Framebuffer{
            .width = opts.width,
            .height = opts.height,
            .color_format = opts.color_format,
            .depth_format = opts.depth_format,
            .sample_count = opts.sample_count,
            .color_img = color_img,
            .color_view = color_view,
            .sample_view = undefined,
        };

        // Depth (matches the color sample count so the pass is consistent).
        if (opts.depth_format != .NONE) {
            fb.depth_img = sg.makeImage(.{
                .width = w,
                .height = h,
                .pixel_format = opts.depth_format,
                .sample_count = @intCast(opts.sample_count),
                .usage = .{ .depth_stencil_attachment = true },
            });
            fb.depth_view = sg.makeView(.{ .depth_stencil_attachment = .{ .image = fb.depth_img } });
        }

        if (msaa) {
            // Single-sampled resolve image is the sampleable result.
            fb.resolve_img = sg.makeImage(.{
                .width = w,
                .height = h,
                .pixel_format = opts.color_format,
                .sample_count = 1,
                .usage = .{ .color_attachment = true, .resolve_attachment = true },
            });
            fb.resolve_view = sg.makeView(.{ .resolve_attachment = .{ .image = fb.resolve_img } });
            fb.sample_view = sg.makeView(.{ .texture = .{ .image = fb.resolve_img } });
        } else {
            // No MSAA so color image is sampled directly.
            fb.sample_view = sg.makeView(.{ .texture = .{ .image = color_img } });
        }

        return fb;
    }

    pub fn deinit(self: *Framebuffer) void {
        sg.destroyView(self.sample_view);
        if (self.sample_count > 1) {
            sg.destroyView(self.resolve_view);
            sg.destroyImage(self.resolve_img);
        }
        if (self.depth_img.id != 0) {
            sg.destroyView(self.depth_view);
            sg.destroyImage(self.depth_img);
        }
        sg.destroyView(self.color_view);
        sg.destroyImage(self.color_img);
    }

    /// A Texture wrapper around the sampleable result, for drawing the FBO's
    /// output through the SpriteBatch (post-processing).
    pub fn asTexture(self: Framebuffer) Texture {
        return .{
            .img = if (self.sample_count > 1) self.resolve_img else self.color_img,
            .view = self.sample_view,
            .width = self.width,
            .height = self.height,
            .format = self.color_format,
        };
    }

    /// The signature batchers/renderers pass to begin() so their pipelines match
    /// this target's formats + sample count.
    pub fn passSignature(self: Framebuffer) PassSignature {
        var sig = PassSignature{
            .color_count = 1,
            .depth_format = self.depth_format,
            .sample_count = self.sample_count,
        };
        sig.color_formats[0] = self.color_format;
        return sig;
    }

    /// This framebuffer's signature, with the depth format overridden — for passes
    /// that attach an EXTERNAL depth buffer (deferred sub-passes use the G-buffer's,
    /// while this framebuffer itself declares .NONE in deferred mode).
    pub fn passSignatureWith(self: Framebuffer, depth_format: sg.PixelFormat) PassSignature {
        var sig = self.passSignature();
        sig.depth_format = depth_format;
        return sig;
    }

    /// Build the sokol Pass that targets this framebuffer. When MSAA, the
    /// resolve attachment is wired so sokol resolves at endPass automatically.
    pub fn pass(self: Framebuffer, action: sg.PassAction) sg.Pass {
        return self.passWith(.{ .action = action });
    }

    /// Build the sokol Pass for this framebuffer, with optional depth override and
    /// resolve control. Handles the MSAA resolve attachment so callers can't forget.
    pub fn passWith(self: Framebuffer, opts: PassOptions) sg.Pass {
        var att = sg.Attachments{};
        att.colors[0] = self.color_view;
        if (self.sample_count > 1 and opts.resolve) att.resolves[0] = self.resolve_view;

        if (opts.depth_view) |dv| {
            att.depth_stencil = dv;
        } else if (self.depth_img.id != 0) {
            att.depth_stencil = self.depth_view;
        }

        return .{ .action = opts.action, .attachments = att };
    }
};
