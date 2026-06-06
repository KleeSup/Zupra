//! src/render/font.zig
//!
//! SDF text rendering.
//!   * Glyphs baked as SIGNED DISTANCE FIELDS into a single-channel (R8)
//!     dynamic atlas.
//!   * Bake-on-demand shelf packer, bounds-safe when the atlas fills.
//!   * Proper baseline layout via stored ascent.
//!   * Atlas uploaded at most ONCE per frame (sokol's rule), frame-guarded.
//!
//! The built-in SDF text shader is owned by the framework and created lazily on
//! first Font creation, therefore the user never touches sokol or the shader.
//!
//! Two ways to draw:
//!   * Low-level: into a batch span you manage, with font.shader bound:
//!       batch.beginEx(cam, .alpha, .{}, font.shader, null);
//!       font.draw(&batch, "Hi", 100, 100, 1.0, zupra.colors.WHITE);
//!       batch.end();
//!   * Convenience: TextBatch2D wraps the span for you:
//!       ```var tb = TextBatch2D.init(&font, &batch);
//!       tb.begin(cam); tb.draw("Hi", 100, 100, 1.0, color); tb.end();```

const std = @import("std");
const sg = @import("sokol").gfx;
const app = @import("sokol").app;
const zupra = @import("../root.zig");

const gfx = @import("../graphics/graphics.zig");
const tex = @import("../graphics/texture.zig");
const math = @import("../math.zig");
const SpriteBatch = @import("spritebatch.zig").SpriteBatch;
const Camera2D = @import("camera.zig").Camera2D;

const stb = zupra.intern.ttf; // vendored stb_truetype

const shd = @import("shaders").sdf;

const Texture = tex.Texture;
const Sprite = tex.Sprite;
const Rect = math.Rect;
const Color = zupra.Color;

// SDF bake parameters: a contract with text_sdf.glsl (edge at onedge/255 ~= 0.5).
const sdf_padding: i32 = 4;
const sdf_onedge: u8 = 128;
const sdf_pixel_dist_scale: f32 = @as(f32, @floatFromInt(sdf_onedge)) / @as(f32, @floatFromInt(sdf_padding));

// --- framework-owned shared text shader ---

var shared_shader: ?gfx.ShaderProgram = null;

/// The built-in SDF text shader. Created on first call (safe once sg.setup has
/// run, which it has by the time a Font is created in user init()).
pub fn sharedShader() gfx.ShaderProgram {
    if (shared_shader == null) {
        shared_shader = gfx.ShaderProgram.init(shd.textSdfShaderDesc, .{
            .layout = .sprite,
            .slots = .{
                .tex_view = shd.VIEW_tex,
                .sampler = shd.SMP_smp,
                .vs_params = shd.UB_vs_params,
            },
        });
    }
    return shared_shader.?;
}

/// Free the shared text shader.
pub fn deinitShared() void {
    if (shared_shader) |*s| {
        s.deinit();
        shared_shader = null;
    }
}

pub const Glyph = struct {
    source: Rect, // glyph rect in the atlas (bake-pixel space)
    x_offset: f32, // from cursor to glyph left (includes -padding)
    y_offset: f32, // from baseline to glyph top (includes -padding)
    x_advance: f32, // cursor advance (bake-pixel space)
};

pub const FontOptions = struct {
    size: f32 = 48,
    atlas_width: u32 = 1024,
    atlas_height: u32 = 1024,
    prebake_ascii: bool = true,
};

pub const Font = struct {
    allocator: std.mem.Allocator,
    shader: gfx.ShaderProgram, // = sharedShader(); user binds this for spans

    texture: Texture, // R8 dynamic atlas
    pixels: []u8, // CPU mirror, 1 byte/pixel
    glyphs: std.AutoHashMap(u32, Glyph),

    font_info: stb.stbtt_fontinfo,
    ttf_buffer: []const u8, // MUST outlive the font: stb holds a pointer into it

    atlas_w: i32,
    atlas_h: i32,
    pen_x: i32 = 0,
    pen_y: i32 = 0,
    row_h: i32 = 0,
    full: bool = false,

    bake_scale: f32,
    ascent: f32,
    line_height: f32,

    dirty: bool = false,
    last_upload_frame: u64 = std.math.maxInt(u64),

    /// `ttf_data` must remain valid for the font's lifetime. No shader argument:
    /// the font uses the framework's built-in SDF text shader.
    pub fn initFromMemory(
        allocator: std.mem.Allocator,
        ttf_data: []const u8,
        opts: FontOptions,
    ) !Font {
        var info: stb.stbtt_fontinfo = undefined;
        if (stb.stbtt_InitFont(&info, ttf_data.ptr, 0) == 0) return error.FontInitFailed;

        const bake_scale = stb.stbtt_ScaleForPixelHeight(&info, opts.size);
        var ascent_i: i32 = 0;
        var descent_i: i32 = 0;
        var line_gap_i: i32 = 0;
        stb.stbtt_GetFontVMetrics(&info, &ascent_i, &descent_i, &line_gap_i);

        const pixels = try allocator.alloc(u8, opts.atlas_width * opts.atlas_height);
        errdefer allocator.free(pixels);
        @memset(pixels, 0);

        var font = Font{
            .allocator = allocator,
            .shader = sharedShader(),
            .texture = Texture.initDynamicFormat(opts.atlas_width, opts.atlas_height, .R8),
            .pixels = pixels,
            .glyphs = std.AutoHashMap(u32, Glyph).init(allocator),
            .font_info = info,
            .ttf_buffer = ttf_data,
            .atlas_w = @intCast(opts.atlas_width),
            .atlas_h = @intCast(opts.atlas_height),
            .bake_scale = bake_scale,
            .ascent = @as(f32, @floatFromInt(ascent_i)) * bake_scale,
            .line_height = @as(f32, @floatFromInt(ascent_i - descent_i + line_gap_i)) * bake_scale,
        };

        if (opts.prebake_ascii) {
            var cp: u32 = 32;
            while (cp < 127) : (cp += 1) _ = font.getOrBake(cp) catch {};
        }
        return font;
    }

    pub fn deinit(self: *Font) void {
        self.glyphs.deinit();
        self.texture.deinit();
        self.allocator.free(self.pixels);
    }

    fn getOrBake(self: *Font, codepoint: u32) !*Glyph {
        if (self.glyphs.getPtr(codepoint)) |g| return g;

        var w: i32 = 0;
        var h: i32 = 0;
        var xoff: i32 = 0;
        var yoff: i32 = 0;
        const sdf = stb.stbtt_GetCodepointSDF(
            &self.font_info,
            self.bake_scale,
            @intCast(codepoint),
            sdf_padding,
            sdf_onedge,
            sdf_pixel_dist_scale,
            &w,
            &h,
            &xoff,
            &yoff,
        );
        defer if (sdf != null) stb.stbtt_FreeSDF(sdf, null);

        var advance_i: i32 = 0;
        stb.stbtt_GetCodepointHMetrics(&self.font_info, @intCast(codepoint), &advance_i, null);
        const advance = @as(f32, @floatFromInt(advance_i)) * self.bake_scale;

        if (sdf == null or w <= 0 or h <= 0) {
            const g = try self.glyphs.getOrPut(codepoint);
            g.value_ptr.* = .{ .source = .{ .x = 0, .y = 0, .width = 0, .height = 0 }, .x_offset = 0, .y_offset = 0, .x_advance = advance };
            return g.value_ptr;
        }

        if (self.pen_x + w > self.atlas_w) {
            self.pen_x = 0;
            self.pen_y += self.row_h + 1;
            self.row_h = 0;
        }
        if (self.pen_y + h > self.atlas_h) {
            self.full = true;
            const g = try self.glyphs.getOrPut(codepoint);
            g.value_ptr.* = .{ .source = .{ .x = 0, .y = 0, .width = 0, .height = 0 }, .x_offset = 0, .y_offset = 0, .x_advance = advance };
            return g.value_ptr;
        }

        var yy: usize = 0;
        while (yy < @as(usize, @intCast(h))) : (yy += 1) {
            var xx: usize = 0;
            while (xx < @as(usize, @intCast(w))) : (xx += 1) {
                const src = yy * @as(usize, @intCast(w)) + xx;
                const dx: usize = @intCast(self.pen_x + @as(i32, @intCast(xx)));
                const dy: usize = @intCast(self.pen_y + @as(i32, @intCast(yy)));
                self.pixels[dy * @as(usize, @intCast(self.atlas_w)) + dx] = sdf[src];
            }
        }
        self.dirty = true;

        const g = try self.glyphs.getOrPut(codepoint);
        g.value_ptr.* = .{
            .source = .{
                .x = @floatFromInt(self.pen_x),
                .y = @floatFromInt(self.pen_y),
                .width = @floatFromInt(w),
                .height = @floatFromInt(h),
            },
            .x_offset = @floatFromInt(xoff),
            .y_offset = @floatFromInt(yoff),
            .x_advance = advance,
        };

        self.pen_x += w + 1;
        if (h > self.row_h) self.row_h = h;
        return g.value_ptr;
    }

    fn uploadIfNeeded(self: *Font) void {
        if (!self.dirty) return;
        const frame = app.frameCount();
        if (frame == self.last_upload_frame) return;
        self.texture.update(self.pixels);
        self.last_upload_frame = frame;
        self.dirty = false;
    }

    pub fn draw(self: *Font, batch: *SpriteBatch, text: []const u8, x: f32, y: f32, scale: f32, color: Color) void {
        var cursor_x = x;
        var cursor_y = y;

        var view = std.unicode.Utf8View.init(text) catch return;
        var it = view.iterator();
        while (it.nextCodepoint()) |cp| {
            if (cp == '\n') {
                cursor_x = x;
                cursor_y += self.line_height * scale;
                continue;
            }

            const glyph = self.getOrBake(cp) catch continue;

            if (glyph.source.width > 0) {
                var sprite = Sprite.init(.{ .texture = self.texture, .source = glyph.source });
                sprite.dest = .{
                    .x = cursor_x + glyph.x_offset * scale,
                    .y = cursor_y + (self.ascent + glyph.y_offset) * scale,
                    .width = glyph.source.width * scale,
                    .height = glyph.source.height * scale,
                };
                sprite.color = color;
                batch.draw(sprite);
            }
            cursor_x += glyph.x_advance * scale;
        }

        self.uploadIfNeeded();
    }

    pub const TextMetrics = struct { width: f32, height: f32 };

    pub fn measure(self: *Font, text: []const u8, scale: f32) TextMetrics {
        var max_w: f32 = 0;
        var cur_w: f32 = 0;
        var lines: f32 = 1;

        var view = std.unicode.Utf8View.init(text) catch return .{ .width = 0, .height = 0 };
        var it = view.iterator();
        while (it.nextCodepoint()) |cp| {
            if (cp == '\n') {
                if (cur_w > max_w) max_w = cur_w;
                cur_w = 0;
                lines += 1;
                continue;
            }
            const glyph = self.getOrBake(cp) catch continue;
            cur_w += glyph.x_advance * scale;
        }
        if (cur_w > max_w) max_w = cur_w;
        return .{ .width = max_w, .height = lines * self.line_height * scale };
    }

    pub fn drawCentered(self: *Font, batch: *SpriteBatch, text: []const u8, cx: f32, cy: f32, scale: f32, color: Color) void {
        const m = self.measure(text, scale);
        self.draw(batch, text, cx - m.width * 0.5, cy - m.height * 0.5, scale, color);
    }
};

/// Convenience wrapper: binds the font's SDF shader and owns the batch span so
/// callers just begin/draw/end without touching the shader. Use this for pure
/// text spans; use Font.draw directly when interleaving with custom batch state.
pub const TextBatch2D = struct {
    font: *Font,
    batch: *SpriteBatch,

    pub fn init(font: *Font, batch: *SpriteBatch) TextBatch2D {
        return .{ .font = font, .batch = batch };
    }

    pub fn begin(self: *TextBatch2D, camera: Camera2D) void {
        self.batch.beginEx(camera, .alpha, .{}, self.font.shader, null);
    }

    pub fn draw(self: *TextBatch2D, text: []const u8, x: f32, y: f32, scale: f32, color: Color) void {
        self.font.draw(self.batch, text, x, y, scale, color);
    }

    pub fn drawCentered(self: *TextBatch2D, text: []const u8, cx: f32, cy: f32, scale: f32, color: Color) void {
        self.font.drawCentered(self.batch, text, cx, cy, scale, color);
    }

    pub fn end(self: *TextBatch2D) void {
        self.batch.end();
    }
};
