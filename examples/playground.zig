const std = @import("std");
const zupra = @import("zupra");

pub fn main(ctx: std.process.Init) !void {
    zupra.init(ctx, .{
        .initFn = init,
        .renderFn = render,
    });
}

pub fn init() void {
    zupra.log.info("Hello from Zupra!", .{});
}

pub fn render() void {
    zupra.beginDrawingClear(zupra.colors.SKYBLUE);
    zupra.endDrawing();
}
