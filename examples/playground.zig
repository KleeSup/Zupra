//! examples/playground.zig
//!
//! 3D render test: a spinning, directionally-lit cube.
//! Geometry is hand-authored (24 verts, per-face normals) so we can test the
//! mesh path before meshbuilder exists.

const std = @import("std");
const zupra = @import("zupra");
const sokol = zupra.intern.sokol;
const zm = zupra.math.zm;

const Vertex3D = zupra.graphics.Vertex3D;
const IndexData = zupra.graphics.IndexData;

var gpa: std.mem.Allocator = undefined;
var cache: zupra.graphics.pipeline.PipelineCache = undefined;
var renderer: zupra.render.MeshRenderer = undefined;
var cam: zupra.render.Camera3D = undefined;
var cube: zupra.render.Mesh = undefined;

var last_ticks: u64 = 0;
var t: f32 = 0;

// Unit cube, 4 verts per face with that face's normal (flat shading).
const s: f32 = 0.5;
const cube_verts = [_]Vertex3D{
    // +X
    .{ .pos = .{ s, -s, -s }, .normal = .{ 1, 0, 0 }, .uv = .{ 0, 0 } },
    .{ .pos = .{ s, s, -s }, .normal = .{ 1, 0, 0 }, .uv = .{ 0, 1 } },
    .{ .pos = .{ s, s, s }, .normal = .{ 1, 0, 0 }, .uv = .{ 1, 1 } },
    .{ .pos = .{ s, -s, s }, .normal = .{ 1, 0, 0 }, .uv = .{ 1, 0 } },
    // -X
    .{ .pos = .{ -s, -s, s }, .normal = .{ -1, 0, 0 }, .uv = .{ 0, 0 } },
    .{ .pos = .{ -s, s, s }, .normal = .{ -1, 0, 0 }, .uv = .{ 0, 1 } },
    .{ .pos = .{ -s, s, -s }, .normal = .{ -1, 0, 0 }, .uv = .{ 1, 1 } },
    .{ .pos = .{ -s, -s, -s }, .normal = .{ -1, 0, 0 }, .uv = .{ 1, 0 } },
    // +Y
    .{ .pos = .{ -s, s, -s }, .normal = .{ 0, 1, 0 }, .uv = .{ 0, 0 } },
    .{ .pos = .{ -s, s, s }, .normal = .{ 0, 1, 0 }, .uv = .{ 0, 1 } },
    .{ .pos = .{ s, s, s }, .normal = .{ 0, 1, 0 }, .uv = .{ 1, 1 } },
    .{ .pos = .{ s, s, -s }, .normal = .{ 0, 1, 0 }, .uv = .{ 1, 0 } },
    // -Y
    .{ .pos = .{ -s, -s, s }, .normal = .{ 0, -1, 0 }, .uv = .{ 0, 0 } },
    .{ .pos = .{ -s, -s, -s }, .normal = .{ 0, -1, 0 }, .uv = .{ 0, 1 } },
    .{ .pos = .{ s, -s, -s }, .normal = .{ 0, -1, 0 }, .uv = .{ 1, 1 } },
    .{ .pos = .{ s, -s, s }, .normal = .{ 0, -1, 0 }, .uv = .{ 1, 0 } },
    // +Z
    .{ .pos = .{ s, -s, s }, .normal = .{ 0, 0, 1 }, .uv = .{ 0, 0 } },
    .{ .pos = .{ s, s, s }, .normal = .{ 0, 0, 1 }, .uv = .{ 0, 1 } },
    .{ .pos = .{ -s, s, s }, .normal = .{ 0, 0, 1 }, .uv = .{ 1, 1 } },
    .{ .pos = .{ -s, -s, s }, .normal = .{ 0, 0, 1 }, .uv = .{ 1, 0 } },
    // -Z
    .{ .pos = .{ -s, -s, -s }, .normal = .{ 0, 0, -1 }, .uv = .{ 0, 0 } },
    .{ .pos = .{ -s, s, -s }, .normal = .{ 0, 0, -1 }, .uv = .{ 0, 1 } },
    .{ .pos = .{ s, s, -s }, .normal = .{ 0, 0, -1 }, .uv = .{ 1, 1 } },
    .{ .pos = .{ s, -s, -s }, .normal = .{ 0, 0, -1 }, .uv = .{ 1, 0 } },
};

var cube_indices = [_]u32{
    0, 1, 2, 0, 2, 3, // +X
    4, 5, 6, 4, 6, 7, // -X
    8, 9, 10, 8, 10, 11, // +Y
    12, 13, 14, 12, 14, 15, // -Y
    16, 17, 18, 16, 18, 19, // +Z
    20, 21, 22, 20, 22, 23, // -Z
};

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
    renderer = zupra.render.MeshRenderer.init(&cache);

    cam = zupra.render.Camera3D.init(16.0 / 9.0);
    cam.position = .{ .x = 3, .y = 2.5, .z = -5 };
    cam.target = .{ .x = 0, .y = 0, .z = 0 };

    cube = zupra.render.MeshBuilder.cube(3, 3, 3); //zupra.render.Mesh.init(&cube_verts, .{ .u32 = &cube_indices });

    last_ticks = sokol.time.now();
}

pub fn render() void {
    const now = sokol.time.now();
    t += @floatCast(sokol.time.sec(sokol.time.diff(now, last_ticks)));
    last_ticks = now;

    // Keep aspect correct on resize / high-DPI (same lesson as the 2D camera).
    cam.setViewport(sokol.app.widthf(), sokol.app.heightf());

    // Tumble: rotate around Y and X over time.
    const model = zm.mul(zm.rotationX(t * 0.6), zm.rotationY(t));

    const material = zupra.render.Material{ .base_color = .{ .r = 0.85, .g = 0.45, .b = 0.2, .a = 1 } };

    zupra.beginDrawingClear(.{ .r = 0.05, .g = 0.06, .b = 0.08, .a = 1 });
    renderer.begin(cam, .{}); // default directional light
    renderer.draw(cube, model, material);
    renderer.end();
    zupra.endDrawing();
}

pub fn deinit() void {
    cube.deinit();
    cache.deinit();
}
