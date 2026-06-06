const std = @import("std");
const sg = @import("sokol").gfx;
const zstbi = @import("zstbi");
const math = @import("../math.zig");
const Rect = math.Rect;
const Vec2 = math.Vec2;
const Color = @import("../root.zig").Color;

pub const Texture = struct {
    img: sg.Image,
    view: sg.View, // texture (sampling) view
    width: u32,
    height: u32,
    format: sg.PixelFormat,

    // --- construction ---

    /// Wrap an already-created image and build its sampling view.
    pub fn init(img: sg.Image, width: u32, height: u32, format: sg.PixelFormat) Texture {
        return .{
            .img = img,
            .view = sg.makeView(.{ .texture = .{ .image = img } }),
            .width = width,
            .height = height,
            .format = format,
        };
    }

    /// Load an encoded image (PNG/JPG/...) from memory. Immutable on the GPU.
    pub fn initBuffer(buffer: []const u8) !Texture {
        var image = try zstbi.Image.loadFromMemory(buffer, 4);
        defer image.deinit(); // safe: sokol copies immutable data during makeImage

        var desc = sg.ImageDesc{
            .width = @intCast(image.width),
            .height = @intCast(image.height),
            .pixel_format = .RGBA8,
            .usage = .{ .immutable = true },
        };
        desc.data.mip_levels[0] = .{ .ptr = image.data.ptr, .size = image.data.len };

        const img = sg.makeImage(desc);
        return Texture.init(img, @intCast(desc.width), @intCast(desc.height), .RGBA8);
    }

    /// Build an immutable texture from raw RGBA8 pixels.
    pub fn initRaw(pixels: []const u8, width: u32, height: u32) Texture {
        var desc = sg.ImageDesc{
            .width = @intCast(width),
            .height = @intCast(height),
            .pixel_format = .RGBA8,
            .usage = .{ .immutable = true },
        };
        desc.data.mip_levels[0] = .{ .ptr = pixels.ptr, .size = pixels.len };

        const img = sg.makeImage(desc);
        return Texture.init(img, width, height, .RGBA8);
    }

    /// CPU-updated texture (no initial data, refreshed each frame via update()).
    pub fn initDynamic(width: u32, height: u32) Texture {
        const desc = sg.ImageDesc{
            .width = @intCast(width),
            .height = @intCast(height),
            .pixel_format = .RGBA8,
            .usage = .{ .dynamic_update = true },
        };
        const img = sg.makeImage(desc);
        return Texture.init(img, width, height, .RGBA8);
    }

    /// CPU-updated single-channel texture (e.g. an SDF font atlas). 1 byte/pixel.
    pub fn initDynamicFormat(width: u32, height: u32, format: sg.PixelFormat) Texture {
        const desc = sg.ImageDesc{
            .width = @intCast(width),
            .height = @intCast(height),
            .pixel_format = format,
            .usage = .{ .dynamic_update = true },
        };
        const img = sg.makeImage(desc);
        return Texture.init(img, width, height, format);
    }

    /// GPU render target that can also be sampled. Single-sampled (MSAA targets
    /// aren't directly sampleable and need a resolve step which is handled separately).
    /// `format` lets a deferred G-buffer pick e.g. RGBA16F / RGBA8 per target.
    pub fn initRenderTarget(width: u32, height: u32, format: sg.PixelFormat) Texture {
        const desc = sg.ImageDesc{
            .width = @intCast(width),
            .height = @intCast(height),
            .pixel_format = format,
            .sample_count = 1,
            .usage = .{ .color_attachment = true },
        };
        const img = sg.makeImage(desc);
        return Texture.init(img, width, height, format);
    }

    /// Load a cubemap from six encoded images. Face order is sokol's:
    /// +X, -X, +Y, -Y, +Z, -Z (right, left, top, bottom, front, back).
    pub fn loadCubemapFromMemory(
        alloc: std.mem.Allocator,
        right: []const u8,
        left: []const u8,
        top: []const u8,
        bottom: []const u8,
        front: []const u8,
        back: []const u8,
        width: u32,
        height: u32,
    ) !Texture {
        const face_size = width * height * 4;
        const all_pixels = try alloc.alloc(u8, face_size * 6);
        defer alloc.free(all_pixels); // safe: immutable data is copied by makeImage

        const raw_faces = [6][]const u8{ right, left, top, bottom, front, back };
        for (raw_faces, 0..) |data, i| {
            var img = try zstbi.Image.loadFromMemory(data, 4);
            defer img.deinit();
            @memcpy(all_pixels[face_size * i .. face_size * (i + 1)], img.data);
        }

        var desc = sg.ImageDesc{
            .type = .CUBE,
            .width = @intCast(width),
            .height = @intCast(height),
            .num_slices = 6,
            .pixel_format = .RGBA8,
            .usage = .{ .immutable = true },
        };
        desc.data.mip_levels[0] = .{ .ptr = all_pixels.ptr, .size = all_pixels.len };

        const img = sg.makeImage(desc);
        return Texture.init(img, width, height, .RGBA8);
    }

    // --- Operations ---

    /// Upload new RGBA8 pixels to a texture created with initDynamic.
    /// sokol allows at most one update per image per frame.
    pub fn update(self: Texture, pixels: []const u8) void {
        var data = sg.ImageData{};
        data.mip_levels[0] = .{ .ptr = pixels.ptr, .size = pixels.len };
        sg.updateImage(self.img, data);
    }

    pub fn toSourceRect(self: Texture) Rect {
        return .{
            .x = 0,
            .y = 0,
            .width = @floatFromInt(self.width),
            .height = @floatFromInt(self.height),
        };
    }

    /// Stable id, used by the batcher to detect texture changes (flush trigger).
    pub inline fn id(self: Texture) u32 {
        return self.img.id;
    }

    pub fn deinit(self: Texture) void {
        sg.destroyView(self.view);
        sg.destroyImage(self.img);
    }
};

pub const TextureRegion = struct {
    texture: Texture,
    source: Rect,

    pub fn full(texture: Texture) TextureRegion {
        return .{ .texture = texture, .source = texture.toSourceRect() };
    }
};

pub const Sprite = struct {
    region: TextureRegion,
    dest: Rect,
    color: Color = .{ .r = 1, .g = 1, .b = 1, .a = 1 },
    origin: Vec2 = .{ .x = 0, .y = 0 },
    rotation: f32 = 0, // radians
    flip_x: bool = false,
    flip_y: bool = false,

    pub fn init(region: TextureRegion) Sprite {
        return .{
            .region = region,
            .dest = .{
                .x = 0,
                .y = 0,
                .width = region.source.width,
                .height = region.source.height,
            },
        };
    }

    pub fn setPosition(self: *Sprite, x: f32, y: f32) void {
        self.dest.x = x;
        self.dest.y = y;
    }

    pub fn setPositionV(self: *Sprite, pos: Vec2) void {
        self.dest.x = pos.x;
        self.dest.y = pos.y;
    }

    pub fn setFlip(self: *Sprite, flip_x: bool, flip_y: bool) void {
        self.flip_x = flip_x;
        self.flip_y = flip_y;
    }

    pub fn setOrigin(self: *Sprite, x: f32, y: f32) void {
        self.origin.x = x;
        self.origin.y = y;
    }

    pub fn setSize(self: *Sprite, width: f32, height: f32) void {
        self.dest.width = width;
        self.dest.height = height;
    }

    pub fn setSizeV(self: *Sprite, size: Vec2) void {
        self.dest.width = size.x;
        self.dest.height = size.y;
    }

    pub fn scale(self: *Sprite, scale_x: f32, scale_y: f32) void {
        self.dest.width *= scale_x;
        self.dest.height *= scale_y;
    }

    pub fn scaleV(self: *Sprite, scale_v: Vec2) void {
        self.dest.width *= scale_v.x;
        self.dest.height *= scale_v.y;
    }

    pub fn scaleF(self: *Sprite, scalef: f32) void {
        self.dest.width *= scalef;
        self.dest.height *= scalef;
    }

    pub fn setColor(self: *Sprite, color: Color) void {
        self.color = color;
    }

    pub fn setRotation(self: *Sprite, rotation: f32) void {
        self.rotation = rotation;
    }

    pub fn rotate(self: *Sprite, delta_rotation: f32) void {
        self.rotation += delta_rotation;
    }

    /// Rotate by a quarter turn `times` times (radians).
    pub fn rotate90(self: *Sprite, times: u32) void {
        self.rotation += (std.math.pi * 0.5) * @as(f32, @floatFromInt(times));
    }
};
