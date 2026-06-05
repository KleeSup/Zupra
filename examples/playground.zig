//! examples/fancy/main.zig
//!
//! Demonstrates using a CUSTOM sprite shader with the SpriteBatch.
//! Run with: zig build run-fancy
//!
//! The fancy shader (shaders/sprite_fancy.glsl) is drop-in compatible with the
//! batcher's vertex/uniform contract, so the only "extension" the user does is:
//!   1. write a .glsl honoring the contract,
//!   2. makeShader() from its generated desc,
//!   3. pass it to batch.beginEx(.., custom_shader).
//! No SpriteBatch changes required.

const std = @import("std");
const zupra = @import("zupra");
const sokol = zupra.intern.sokol; // re-exported by the framework

const img_data = @embedFile("assets/icon.png");

var gpa: std.mem.Allocator = undefined;
var cache: zupra.graphics.pipeline.PipelineCache = undefined;
var batch: zupra.render.SpriteBatch = undefined;
var cam: zupra.render.Camera2D = undefined;
var icon: zupra.graphics.texture.Texture = undefined;
var fancy: zupra.graphics.ShaderProgram = undefined;

var last_ticks: u64 = 0;
var t: f32 = 0;

pub fn main(ctx: std.process.Init) !void {
    gpa = ctx.gpa;
    zupra.init(ctx, .{
        .initFn = init,
        .renderFn = render,
        .deinitFn = deinit,
    });
}

pub fn init() void {
    zupra.log.info("Fancy shader example", .{});
    cache = .init(gpa);
    batch = zupra.render.SpriteBatch.init(gpa, &cache, .{}) catch unreachable;
    cam = .init(1280, 720);
    icon = zupra.graphics.texture.Texture.initBuffer(img_data) catch unreachable;

    // Compile-time-generated desc for shaders/sprite_fancy.glsl.
    fancy = zupra.graphics.ShaderProgram.init(@import("assets/shaders/sprite_fancy.glsl.zig").spriteFancyShaderDesc, .{});

    last_ticks = sokol.time.now();
}

pub fn render() void {
    // Frame-rate independent time (vsync is off by default in Config).
    const now = sokol.time.now();
    t += @floatCast(sokol.time.sec(sokol.time.diff(now, last_ticks)));
    last_ticks = now;

    // Build a sprite fresh each frame and animate its transform on the CPU.
    var sprite = zupra.graphics.texture.Sprite.init(.full(icon));

    const base_w = sprite.dest.width;
    const base_h = sprite.dest.height;
    const pulse = 1.0 + 0.25 * @sin(t * 3.0);
    sprite.setSize(base_w * pulse, base_h * pulse);
    sprite.setOrigin(sprite.dest.width * 0.5, sprite.dest.height * 0.5); // spin about center

    const cx: f32 = 640.0;
    const cy: f32 = 360.0;
    const orbit: f32 = 140.0;
    sprite.setPosition(cx + @cos(t) * orbit, cy + @sin(t) * orbit);
    sprite.setRotation(t * 1.5);

    zupra.beginDrawingClear(zupra.colors.BLACK);

    // The whole point: hand the batch our custom shader. PassSignature defaults
    // to the swapchain (.{}). Everything else is the normal batcher path.
    batch.beginEx(cam, .alpha, .{}, fancy);
    batch.draw(sprite);
    batch.end();

    zupra.endDrawing();
}

pub fn deinit() void {
    fancy.deinit();
    icon.deinit();
    batch.deinit(gpa);
    cache.deinit();
}
