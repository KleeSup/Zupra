pub const sokol = @import("sokol");
pub const zstbi = @import("zstbi");
pub const zmesh = @import("zmesh");
pub const zmath = @import("zmath");
pub const ttf = @cImport({
    @cInclude("stb_truetype.h");
});
pub const debug_views = @import("build_options").debug_views;

const graphics = @import("graphics/graphics.zig");
const zupra = @import("root.zig");
// --- Internals ---

pub var pipeline_cache: graphics.pipeline.PipelineCache = undefined;
pub var white_1x1: graphics.texture.Texture = undefined;

var initialized = false;
pub fn init() *const fn () void {
    if (initialized) @panic("zupra.intern.init() called! This should never be invoked from outside the framework!");
    pipeline_cache = graphics.pipeline.PipelineCache.init(zupra.getGPA());
    const a: [4]u8 = .{ 255, 255, 255, 255 };
    white_1x1 = graphics.texture.Texture.initRaw(&a, 1, 1);

    initialized = true;
    return deinit;
}

fn deinit() void {
    pipeline_cache.deinit();
    white_1x1.deinit();
}
