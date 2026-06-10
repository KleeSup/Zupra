//! examples/playground.zig
//!
//! Model render test: a Model (single cube submesh + material) drawn through
//! ModelBatch as a spinning ModelInstance.

const std = @import("std");
const zupra = @import("zupra");
const sokol = zupra.intern.sokol;

var gpa: std.mem.Allocator = undefined;
var cache: zupra.graphics.pipeline.PipelineCache = undefined;
var mbatch: zupra.render.ModelBatch = undefined;
var cam: zupra.render.Camera3D = undefined;
var model: zupra.render.Model = undefined;

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
    cache = .init(gpa);
    mbatch = zupra.render.ModelBatch.init(&cache);

    cam = zupra.render.Camera3D.init(16.0 / 9.0);
    cam.position = .{ .x = 3, .y = 2.5, .z = -5 };
    cam.target = .{ .x = 0, .y = 0, .z = 0 };

    const cube = zupra.render.MeshBuilder.sphere(gpa, 1, 1, 1) catch unreachable;
    model = zupra.render.Model.fromMesh(gpa, cube, .{
        .base_color = .{ .r = 0.85, .g = 0.45, .b = 0.2, .a = 1 },
    }) catch unreachable;

    last_ticks = sokol.time.now();
}

pub fn render() void {
    const now = sokol.time.now();
    t += @floatCast(sokol.time.sec(sokol.time.diff(now, last_ticks)));
    last_ticks = now;

    cam.setViewport(sokol.app.widthf(), sokol.app.heightf());

    // Place an instance, spin it about the Y axis.
    var inst = model.instance();
    inst.setRotationAxisAngle(.{ .x = 0, .y = 1, .z = 0 }, t);

    zupra.beginDrawingClear(.{ .r = 0.05, .g = 0.06, .b = 0.08, .a = 1 });
    mbatch.begin(cam);
    mbatch.draw(inst);
    mbatch.end();
    zupra.endDrawing();
}

pub fn deinit() void {
    model.deinit();
    cache.deinit();
}
