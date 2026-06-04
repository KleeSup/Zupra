const std = @import("std");
const zupra = @import("zupra");

const img_data = @embedFile("assets/icon.png");
var icon: zupra.graphics.texture.Texture = undefined;

var gpa: std.mem.Allocator = undefined;
var cache: zupra.graphics.pipeline.PipelineCache = undefined;
var batch: zupra.render.SpriteBatch = undefined;
var cam: zupra.render.Camera2D = undefined;

pub fn main(ctx: std.process.Init) !void {
    gpa = ctx.gpa;
    zupra.init(ctx, .{
        .initFn = init,
        .renderFn = render,
        .deinitFn = deinit,
    });
}

pub fn init() void {
    zupra.log.info("Hello from Zupra!", .{});
    cache = .init(gpa);
    batch = zupra.render.SpriteBatch.init(gpa, &cache, .{}) catch unreachable;
    cam = .init(1280, 720);
    icon = zupra.graphics.texture.Texture.initBuffer(img_data) catch unreachable;
}

pub fn render() void {
    zupra.beginDrawingClear(zupra.colors.SKYBLUE);
    batch.begin(cam, .alpha);

    var reg = zupra.graphics.texture.Sprite.init(.full(icon));
    reg.dest.width *= 0.5;
    reg.dest.height *= 0.5;
    batch.draw(reg);

    batch.end();
    zupra.endDrawing();
}

pub fn deinit() void {
    batch.deinit(gpa);
    cache.deinit();
    icon.deinit();
}
