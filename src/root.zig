pub const math = @import("math.zig");
pub const intern = @import("intern.zig");
pub const log = @import("util/log.zig");

const sokol = @import("sokol");
const std = @import("std");
const zstbi = @import("zstbi");
const zmesh = @import("zmesh");

// --- Color ---

pub const Color = sokol.gfx.Color;
pub const colors = struct {
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
};

// --- Drawing ---

var drawing: bool = false;
var fb_drawing: bool = false;
pub fn isDrawingToFrameBuffer() bool {
    return fb_drawing;
}

pub fn beginDrawingPass(pass: sokol.gfx.Pass) void {
    if (drawing) @panic("endDrawing() needs to be called before beginDrawing()!");
    drawing = true;
    sokol.gfx.beginPass(pass);
}
pub fn beginDrawingClear(clear_color: Color) void {
    beginDrawingPass(.{ .action = buildClearAction(clear_color), .swapchain = sokol.glue.swapchain() });
}
pub fn beginDrawing() void {
    beginDrawingPass(.{ .swapchain = sokol.glue.swapchain() });
}

pub fn endDrawing() void {
    if (!drawing) @panic("beginDrawing() needs to be called before endDrawing()!");
    drawing = false;
    fb_drawing = false;
    sokol.gfx.endPass();
}

pub fn buildClearAction(clear_color: Color) sokol.gfx.PassAction {
    var frame_action = sokol.gfx.PassAction{};
    frame_action.colors[0] = .{ .load_action = .CLEAR, .clear_value = clear_color };
    frame_action.depth = .{ .load_action = .CLEAR, .clear_value = 1.0 };
    return frame_action;
}

// --- Management ---

pub fn quit() void {
    sokol.app.requestQuit();
}

pub fn exit() void {
    sokol.app.quit();
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
    sample_count: u8 = 4, //4 for MSAAx4
    pools: PoolSetup = .{},

    initFn: *const fn () void,
    renderFn: *const fn () void,
    eventFn: ?*const fn (event: *const sokol.app.Event) void = null,
    deinitFn: ?*const fn () void = null,
};
var user_config: Config = undefined;

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
    user_config.initFn();
}
export fn _deinit() void {
    if (user_config.deinitFn) |deinitFn| {
        deinitFn();
    }
    zstbi.deinit();
    zmesh.deinit();
}

export fn _render() void {
    user_config.renderFn();
}
export fn _event(event: [*c]const sokol.app.Event) callconv(.c) void {
    if (user_config.eventFn) |eventFn| {
        eventFn(event);
    }
}
