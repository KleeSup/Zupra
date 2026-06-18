//! examples/playground.zig
//!
//! Full pipeline demo via SceneRenderer (deferred): PBR opaque geometry under a
//! directional + point light, with a transparent sphere sweeping across in front
//! to show sorted, depth-correct transparency. Switch .deferred -> .forward and
//! everything still renders (Lambert instead of PBR), no other code change.

const std = @import("std");
const zupra = @import("zupra");
const sokol = zupra.intern.sokol;

const Model = zupra.render.Model;
const Material = zupra.render.Material;
const MeshBuilder = zupra.render.MeshBuilder;
const Light = zupra.render.Light;
const Environment = zupra.render.Environment;

var gpa: std.mem.Allocator = undefined;
var cache: zupra.graphics.pipeline.PipelineCache = undefined;
var scene: zupra.render.SceneRenderer = undefined;
var cam: zupra.render.Camera3D = undefined;
var env: Environment = .{};

var floor_model: Model = undefined;
var metal_model: Model = undefined;
var rough_model: Model = undefined;
var glass_model: Model = undefined;

var last_ticks: u64 = 0;
var t: f32 = 0;

pub fn main(ctx: std.process.Init) !void {
    gpa = ctx.gpa;
    zupra.init(ctx, .{ .initFn = init, .renderFn = render, .deinitFn = deinit });
}

pub fn init() void {
    cache = .init(gpa);
    scene = zupra.render.SceneRenderer.init(gpa, &cache, .deferred, 1280, 720);

    cam = zupra.render.Camera3D.init(16.0 / 9.0);

    // Floor: large rough dielectric plane.
    floor_model = Model.fromMesh(gpa, MeshBuilder.plane(24, 24), .{
        .base_color = .{ .r = 0.5, .g = 0.5, .b = 0.55, .a = 1 },
        .metallic = 0.0,
        .roughness = 0.9,
    }) catch unreachable;

    // Polished metal sphere.
    metal_model = Model.fromMesh(gpa, MeshBuilder.sphere(gpa, 0.7, 48, 48) catch unreachable, .{
        .base_color = .{ .r = 1.0, .g = 0.8, .b = 0.4, .a = 1 },
        .metallic = 1.0,
        .roughness = 0.18,
    }) catch unreachable;

    // Rough dielectric sphere.
    rough_model = Model.fromMesh(gpa, MeshBuilder.sphere(gpa, 0.7, 48, 48) catch unreachable, .{
        .base_color = .{ .r = 0.85, .g = 0.25, .b = 0.2, .a = 1 },
        .metallic = 0.0,
        .roughness = 0.65,
    }) catch unreachable;

    // Transparent sphere (routes to the sorted forward transparent pass).
    glass_model = Model.fromMesh(gpa, MeshBuilder.sphere(gpa, 0.8, 48, 48) catch unreachable, .{
        .base_color = .{ .r = 0.4, .g = 0.7, .b = 1.0, .a = 0.4 },
        .metallic = 0.0,
        .roughness = 0.1,
        .alpha_mode = .blend,
    }) catch unreachable;

    // Lighting: a warm sun + a colored point light.
    env = .{ .ambient = .{ .r = 0.03, .g = 0.03, .b = 0.04, .a = 1 } };
    env.addLight(Light.directional(.{ .x = -0.5, .y = -1.0, .z = -0.35 }, .{ .r = 1.0, .g = 0.96, .b = 0.9, .a = 1 }, 2.5));
    env.addLight(Light.point(.{ .x = 2.5, .y = 2.5, .z = 2.0 }, .{ .r = 0.1, .g = 0.1, .b = 0.9, .a = 1 }, 35.0, 14.0));

    last_ticks = sokol.time.now();
}

pub fn render() void {
    const now = sokol.time.now();
    t += @floatCast(sokol.time.sec(sokol.time.diff(now, last_ticks)));
    last_ticks = now;

    // Orbit the camera around the scene.
    const r: f32 = 6.0;
    cam.position = .{ .x = @sin(t * 0.3) * r, .y = 3.0, .z = @cos(t * 0.3) * r };
    cam.target = .{ .x = 0, .y = 0.2, .z = 0 };

    var floor = floor_model.instance();
    floor.setPosition(.{ .x = 0, .y = -0.7, .z = 0 });

    var metal = metal_model.instance();
    metal.setPosition(.{ .x = -1.8, .y = 0, .z = 0 });

    var rough = rough_model.instance();
    rough.setPosition(.{ .x = 0.2, .y = 0, .z = 0 });

    // Glass sweeps left-right in front of the spheres.
    var glass = glass_model.instance();
    glass.setPosition(.{ .x = @sin(t * 0.6) * 2.5, .y = 0.1, .z = 2.2 });

    scene.begin(cam, env);
    scene.draw(floor);
    scene.draw(metal);
    scene.draw(rough);
    scene.draw(glass);
    scene.end();
}

pub fn deinit() void {
    floor_model.deinit();
    metal_model.deinit();
    rough_model.deinit();
    glass_model.deinit();
    scene.deinit();
    cache.deinit();
}
