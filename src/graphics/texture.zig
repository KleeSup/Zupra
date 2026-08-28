const std = @import("std");
const sg = @import("sokol").gfx;
const zstbi = @import("zstbi");
const math = @import("../math.zig");
const Rect = math.Rect;
const Vec2 = math.Vec2;
const Color = @import("../root.zig").Color;

/// How the RGB channels of an 8-bit texture are encoded on disk.
///
/// GPU sampling converts `.srgb` to linear light automatically, while alpha
/// stays linear as required by the glTF and GPU sRGB specifications. Surface
/// colour/emissive maps use `.srgb`; normals, masks and PBR parameters use
/// `.linear` because their numbers are data rather than light intensities.
pub const ColorSpace = enum {
    linear,
    srgb,

    pub fn pixelFormat(self: ColorSpace) sg.PixelFormat {
        return switch (self) {
            .linear => .RGBA8,
            .srgb => .SRGB8A8,
        };
    }
};

// zstbi exposes the linear `stbir_resize_uint8` wrapper, while its bundled C
// implementation also exports this sRGB-aware variant. Calling it directly
// keeps mip filtering in linear light without carrying a second image library.
extern fn stbir_resize_uint8_srgb(
    input_pixels: [*]const u8,
    input_w: c_int,
    input_h: c_int,
    input_stride_in_bytes: c_int,
    output_pixels: [*]u8,
    output_w: c_int,
    output_h: c_int,
    output_stride_in_bytes: c_int,
    num_channels: c_int,
    alpha_channel: c_int,
    flags: c_int,
) c_int;

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

    /// Load an encoded image (PNG/JPG/...) from memory as linear data.
    /// Use `initBufferWithColorSpace(..., .srgb)` for an authored colour image.
    pub fn initBuffer(buffer: []const u8) !Texture {
        return initBufferWithColorSpace(buffer, .linear);
    }

    /// Load an encoded 8-bit image from memory with explicit RGB encoding.
    pub fn initBufferWithColorSpace(buffer: []const u8, color_space: ColorSpace) !Texture {
        try ensureColorSpaceSupported(color_space);
        var image = try zstbi.Image.loadFromMemory(buffer, 4);
        defer image.deinit(); // safe: sokol copies immutable data during makeImage

        if (image.bytes_per_component != 1) return error.UnsupportedColorSpaceFormat;

        var desc = sg.ImageDesc{
            .width = @intCast(image.width),
            .height = @intCast(image.height),
            .pixel_format = color_space.pixelFormat(),
            .usage = .{ .immutable = true },
        };
        desc.data.mip_levels[0] = .{ .ptr = image.data.ptr, .size = image.data.len };

        const img = sg.makeImage(desc);
        return Texture.init(img, @intCast(desc.width), @intCast(desc.height), color_space.pixelFormat());
    }

    /// Load an encoded 8-bit image and upload every supported mip level.
    ///
    /// This is the path for material images, especially glTF assets: their
    /// samplers commonly request a mipmapped minification filter, while sokol
    /// immutable images require every mip to be supplied at creation time.
    /// The temporary levels are CPU-generated with stb_image_resize and freed
    /// after `makeImage`, which copies them into GPU-owned storage.
    pub fn initBufferMipmapped(buffer: []const u8) !Texture {
        return initBufferMipmappedWithColorSpace(buffer, .linear);
    }

    /// Load an encoded 8-bit image and generate every supported mip level.
    ///
    /// For `.srgb`, downsampling happens in linear light and the result is
    /// re-encoded to sRGB before upload. Averaging encoded RGB values directly
    /// produces visibly dark, muddy distant albedo/emissive mips.
    pub fn initBufferMipmappedWithColorSpace(buffer: []const u8, color_space: ColorSpace) !Texture {
        try ensureColorSpaceSupported(color_space);
        // Sokol's ImageData currently exposes 16 mip slots. Keep the temporary
        // storage in lockstep with that limit, rather than allocating per level.
        var mip_images: [16]zstbi.Image = undefined;
        mip_images[0] = try zstbi.Image.loadFromMemory(buffer, 4);
        var mip_count: usize = 1;
        defer {
            for (mip_images[0..mip_count]) |*mip| mip.deinit();
        }

        // `Image.resize` uses stb_image_resize's 8-bit path. glTF's core PNG
        // and JPEG material images are 8-bit; reject formats this path cannot
        // resize correctly instead of silently uploading malformed mip data.
        if (mip_images[0].bytes_per_component != 1) return error.UnsupportedMipFormat;

        while (mip_count < mip_images.len) {
            const previous = &mip_images[mip_count - 1];
            if (previous.width == 1 and previous.height == 1) break;

            // Standard texture mip dimensions are floor(previous / 2), clamped
            // to one texel. This matches the dimensions sokol validates for an
            // immutable mip chain, including non-power-of-two source images.
            const next_width = @max(1, previous.width / 2);
            const next_height = @max(1, previous.height / 2);
            mip_images[mip_count] = switch (color_space) {
                .linear => previous.resize(next_width, next_height),
                .srgb => try resizeSrgb(previous, next_width, next_height),
            };
            mip_count += 1;
        }

        var desc = sg.ImageDesc{
            .width = @intCast(mip_images[0].width),
            .height = @intCast(mip_images[0].height),
            .num_mipmaps = @intCast(mip_count),
            .pixel_format = color_space.pixelFormat(),
            .usage = .{ .immutable = true },
        };
        for (mip_images[0..mip_count], 0..) |mip, level| {
            desc.data.mip_levels[level] = .{ .ptr = mip.data.ptr, .size = mip.data.len };
        }

        const img = sg.makeImage(desc);
        return Texture.init(img, @intCast(desc.width), @intCast(desc.height), color_space.pixelFormat());
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

fn ensureColorSpaceSupported(color_space: ColorSpace) !void {
    if (color_space != .srgb) return;
    const info = sg.queryPixelformat(.SRGB8A8);
    // Material samplers may request linear filtering and mip interpolation, so
    // merely being sampleable is not enough for a useful sRGB material image.
    if (!info.sample or !info.filter) return error.UnsupportedSrgbTexture;
}

fn resizeSrgb(source: *const zstbi.Image, width: u32, height: u32) !zstbi.Image {
    // Images in this path are decoded with `forced_num_components = 4`. The
    // alpha channel remains linear (index 3), and flags=0 asks stb to handle
    // straight-alpha source data correctly when filtering coloured cut-outs.
    std.debug.assert(source.num_components == 4);
    std.debug.assert(source.bytes_per_component == 1);

    var result = try zstbi.Image.createEmpty(width, height, 4, .{});
    errdefer result.deinit();
    const ok = stbir_resize_uint8_srgb(
        source.data.ptr,
        @intCast(source.width),
        @intCast(source.height),
        0,
        result.data.ptr,
        @intCast(width),
        @intCast(height),
        0,
        4,
        3,
        0,
    );
    if (ok == 0) return error.SrgbMipGenerationFailed;
    return result;
}

test "texture color spaces choose their GPU formats" {
    try std.testing.expectEqual(sg.PixelFormat.RGBA8, ColorSpace.linear.pixelFormat());
    try std.testing.expectEqual(sg.PixelFormat.SRGB8A8, ColorSpace.srgb.pixelFormat());
}

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

// -=- Packing -=-

/// Pack separate roughness and metalness JPG/PNG byte blobs into one RGBA8 texture
/// laid out (R=255, G=roughness, B=metallic, A=255).
/// Both inputs must be the same dimensions. `allocator` is used only
/// for scratch because the GPU texture owns its data after upload.
///
/// Example Usage (with @embedFile):
///   const mr = try packMetallicRoughness(gpa,
///       @embedFile("assets/metal/..._Roughness.jpg"),
///       @embedFile("assets/metal/..._Metalness.jpg"));
pub fn packMetallicRoughness(
    allocator: std.mem.Allocator,
    roughness_bytes: []const u8,
    metallic_bytes: []const u8,
) !Texture {

    // Load both as single-channel grayscale.
    var rough = try zstbi.Image.loadFromMemory(roughness_bytes, 1);
    defer rough.deinit();
    var metal = try zstbi.Image.loadFromMemory(metallic_bytes, 1);
    defer metal.deinit();

    if (rough.width != metal.width or rough.height != metal.height) {
        return error.MapSizeMismatch;
    }

    const w = rough.width;
    const h = rough.height;
    const px = w * h;

    // Build RGBA8: R=255, G=roughness, B=metallic, A=255.
    const packed_data = try allocator.alloc(u8, px * 4);
    defer allocator.free(packed_data);

    var i: usize = 0;
    while (i < px) : (i += 1) {
        packed_data[i * 4 + 0] = 255; // R unused (could hold occlusion for ORM)
        packed_data[i * 4 + 1] = rough.data[i]; // G = roughness
        packed_data[i * 4 + 2] = metal.data[i]; // B = metallic
        packed_data[i * 4 + 3] = 255; // A unused
    }

    // Linear RGBA8 (NOT sRGB — this is data, not color).
    return Texture.initRaw(packed_data, @intCast(w), @intCast(h));
}

/// Same as packMetallicRoughness, but also folds an occlusion map into R (glTF "ORM" packing with R=occlusion, G=roughness, B=metallic). Pass all three same-size grayscales.
pub fn packOcclusionRoughnessMetallic(
    allocator: std.mem.Allocator,
    occlusion_bytes: []const u8,
    roughness_bytes: []const u8,
    metallic_bytes: []const u8,
) !Texture {
    var occ = try zstbi.Image.loadFromMemory(occlusion_bytes, 1);
    defer occ.deinit();
    var rough = try zstbi.Image.loadFromMemory(roughness_bytes, 1);
    defer rough.deinit();
    var metal = try zstbi.Image.loadFromMemory(metallic_bytes, 1);
    defer metal.deinit();

    if (rough.width != metal.width or rough.height != metal.height or
        occ.width != rough.width or occ.height != rough.height)
    {
        return error.MapSizeMismatch;
    }

    const w = rough.width;
    const h = rough.height;
    const px = w * h;

    const packed_data = try allocator.alloc(u8, px * 4);
    defer allocator.free(packed_data);

    var i: usize = 0;
    while (i < px) : (i += 1) {
        packed_data[i * 4 + 0] = occ.data[i]; // R = occlusion
        packed_data[i * 4 + 1] = rough.data[i]; // G = roughness
        packed_data[i * 4 + 2] = metal.data[i]; // B = metallic
        packed_data[i * 4 + 3] = 255;
    }

    return Texture.initRaw(packed_data, @intCast(w), @intCast(h));
}
