pub const sokol = @import("sokol");
pub const zstbi = @import("zstbi");
pub const zmesh = @import("zmesh");
pub const zmath = @import("zmath");
pub const ttf = @cImport({
    @cInclude("stb_truetype.h");
});

const graphics = @import("graphics/graphics.zig");
// --- Internals ---
pub const gpa = @import("std").heap.c_allocator;

pub var pipeline_cache: graphics.pipeline.PipelineCache = undefined;

var initialized = false;
pub fn init() *const fn () void {
    if (initialized) @panic("zupra.intern.init() called! This should never be invoked from outside the framework!");
    pipeline_cache = .init(gpa);

    initialized = true;
    return deinit;
}

fn deinit() void {
    pipeline_cache.deinit();
}
