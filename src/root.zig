pub const math = @import("math.zig");
pub const intern = @import("intern.zig");
pub const log = @import("util/log.zig");
pub const render = @import("render/render.zig");
pub const graphics = @import("graphics/graphics.zig");
pub const app = @import("app.zig");
pub const input = @import("input/input.zig");
pub const time = sokol.time;

pub const Event = sokol.app.Event;

const sokol = @import("sokol");
const std = @import("std");
const zstbi = @import("zstbi");
const zmesh = @import("zmesh");

// --- Color ---

pub const Color = sokol.gfx.Color;
pub const colors = opaque {
    pub const BLACK = Color{ .r = 0, .g = 0, .b = 0, .a = 1.0 };
    pub const WHITE = Color{ .r = 1.0, .g = 1.0, .b = 1.0, .a = 1.0 };
    pub const RED = Color{ .r = 1.0, .g = 0, .b = 0, .a = 1.0 };
    pub const GREEN = Color{ .r = 0, .g = 1.0, .b = 0, .a = 1.0 };
    pub const BLUE = Color{ .r = 0, .g = 0, .b = 1.0, .a = 1.0 };
    pub const YELLOW = Color{ .r = 1.0, .g = 1.0, .b = 0, .a = 1.0 };
    pub const CYAN = Color{ .r = 0, .g = 1.0, .b = 1.0, .a = 1.0 };
    pub const MAGENTA = Color{ .r = 1.0, .g = 0, .b = 1.0, .a = 1.0 };
    pub const SKYBLUE = Color{ .r = 0.529, .g = 0.808, .b = 0.922, .a = 1.0 };
    pub const GRAY = Color{ .r = 0.5, .g = 0.5, .b = 0.5, .a = 1.0 };

    /// Pack a normalized RGBA color into a UBYTE4N u32.
    /// `r` is the least-significant byte.
    pub fn packColorF(r: f32, g: f32, b: f32, a: f32) u32 {
        const R: u32 = @intFromFloat(std.math.clamp(r, 0, 1) * 255.0 + 0.5);
        const G: u32 = @intFromFloat(std.math.clamp(g, 0, 1) * 255.0 + 0.5);
        const B: u32 = @intFromFloat(std.math.clamp(b, 0, 1) * 255.0 + 0.5);
        const A: u32 = @intFromFloat(std.math.clamp(a, 0, 1) * 255.0 + 0.5);
        return R | (G << 8) | (B << 16) | (A << 24);
    }

    pub inline fn packColor(color: Color) u32 {
        return packColorF(color.r, color.g, color.b, color.a);
    }
};

// --- Drawing ---

var drawing: bool = false;
var fb_drawing: bool = false;
pub fn isDrawingToFrameBuffer() bool {
    return fb_drawing;
}

/// Begin a drawing pass with the given pass descriptor.
/// This is a low-level function that allows you to specify custom pass descriptors,
/// but you can also use the more convenient beginDrawingClear() and beginDrawing() functions for common use cases.
pub fn beginDrawingPass(pass: sokol.gfx.Pass) void {
    if (drawing) @panic("endDrawing() needs to be called before beginDrawing()!");
    drawing = true;
    sokol.gfx.beginPass(pass);
}

/// Begin a drawing pass with the given clear color. This will automatically clear the screen with the specified color at the beginning of the pass.
pub fn beginDrawingClear(clear_color: Color) void {
    beginDrawingPass(.{ .action = buildClearAction(clear_color), .swapchain = sokol.glue.swapchain() });
}

/// Begin a drawing pass without clearing the screen.
/// This is useful for drawing on top of existing content, but make sure to call endDrawing() when you're done.
pub fn beginDrawing() void {
    beginDrawingPass(.{ .swapchain = sokol.glue.swapchain() });
}

/// End the current drawing pass. This will submit all drawing commands to the GPU and present the frame if you're drawing to the swapchain.
pub fn endDrawing() void {
    if (!drawing) @panic("beginDrawing() needs to be called before endDrawing()!");
    drawing = false;
    fb_drawing = false;
    sokol.gfx.endPass();
}

/// Builds a pass action that clears the screen with the specified color.
/// This is a helper function that can be used with beginDrawingPass() to easily clear the screen at the beginning of a drawing pass.
pub fn buildClearAction(clear_color: Color) sokol.gfx.PassAction {
    var frame_action = sokol.gfx.PassAction{};
    frame_action.colors[0] = .{ .load_action = .CLEAR, .clear_value = clear_color };
    frame_action.depth = .{ .load_action = .CLEAR, .clear_value = 1.0 };
    return frame_action;
}

// --- Init ---

pub const PoolSetup = struct {
    buffer_pool_size: i32 = 0,
    image_pool_size: i32 = 0,
    sampler_pool_size: i32 = 0,
    shader_pool_size: i32 = 0,
    pipeline_pool_size: i32 = 0,
    view_pool_size: i32 = 0,
    uniform_buffer_size: i32 = 0,
};

pub const Config = struct {
    title: [:0]const u8 = "Zupra Framework",
    width: u32 = 1280,
    height: u32 = 720,
    fullscreen: bool = false,
    v_sync: bool = false,
    sample_count: u8 = 1, //4 for MSAAx4
    pools: PoolSetup = .{},
    index_type: graphics.IndexType = .u32,

    initFn: *const fn () void,
    renderFn: *const fn () void,
    eventFn: ?*const fn (event: *const sokol.app.Event) void = null,
    deinitFn: ?*const fn () void = null,
};
var user_config: Config = undefined;
var intern_deinit: *const fn () void = undefined;

pub fn init(ctx: std.process.Init, config: Config) void {
    user_config = config;
    const scfg = sokol.app.Desc{
        .window_title = user_config.title,
        .width = @intCast(user_config.width),
        .height = @intCast(user_config.height),
        .fullscreen = user_config.fullscreen,
        .sample_count = user_config.sample_count,
        .swap_interval = if (config.v_sync) 1 else 0,
        .init_cb = _init,
        .frame_cb = _render,
        .event_cb = _event,
        .cleanup_cb = _deinit,
        .logger = .{ .func = sokol.log.func },
    };

    zstbi.init(ctx.io, std.heap.c_allocator);
    zmesh.init(std.heap.c_allocator);

    sokol.app.run(scfg);
}

export fn _init() void {
    sokol.gfx.setup(.{
        .environment = sokol.glue.environment(),
        .logger = .{ .func = sokol.log.func },
        .buffer_pool_size = user_config.pools.buffer_pool_size,
        .image_pool_size = user_config.pools.image_pool_size,
        .sampler_pool_size = user_config.pools.sampler_pool_size,
        .shader_pool_size = user_config.pools.shader_pool_size,
        .pipeline_pool_size = user_config.pools.pipeline_pool_size,
        .view_pool_size = user_config.pools.view_pool_size,
        .uniform_buffer_size = user_config.pools.uniform_buffer_size,
    });
    sokol.time.setup();
    intern_deinit = intern.init();
    user_config.initFn();
}
export fn _deinit() void {
    if (user_config.deinitFn) |deinitFn| {
        deinitFn();
    }
    zstbi.deinit();
    zmesh.deinit();
    intern_deinit();
    sokol.gfx.shutdown();
}

export fn _render() void {
    input._updateFrame();
    user_config.renderFn();
    sokol.gfx.commit();
}
export fn _event(event: [*c]const sokol.app.Event) callconv(.c) void {
    input._updateEvent(event);
    if (user_config.eventFn) |eventFn| {
        eventFn(event);
    }
}
