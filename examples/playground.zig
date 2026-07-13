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
const Texture = zupra.graphics.texture.Texture;

var gpa: std.mem.Allocator = undefined;
var cache: zupra.graphics.pipeline.PipelineCache = undefined;
var scene: zupra.render.SceneRenderer = undefined;
var cam: zupra.render.Camera3D = undefined;
var env: Environment = .{};
var controller: zupra.render.FirstPersonController = undefined;

var floor_model: Model = undefined;
var metal_model: Model = undefined;
var rough_model: Model = undefined;
var glass_model: Model = undefined;
var wall_model: Model = undefined;
var helmet_model: Model = undefined;

var model_queue: std.ArrayList(Model) = .empty;

var last_ticks: u64 = 0;
var t: f32 = 0;

pub fn main(ctx: std.process.Init) !void {
    gpa = ctx.gpa;
    zupra.init(ctx, .{ .initFn = init, .renderFn = render, .deinitFn = deinit, .eventFn = onEvent });
}

pub fn init() void {
    cache = .init(gpa);
    scene = zupra.render.SceneRenderer.init(gpa, &cache, .forward, 1280, 720);

    cam = zupra.render.Camera3D.init(16.0 / 9.0);
    controller = .init(.{ .x = 0, .y = 0, .z = 0 });

    const bricks = zupra.graphics.texture.Texture.initBuffer(@embedFile("assets/bricks/Bricks097_1K-JPG_Color.jpg")) catch unreachable;
    const bricks_normal = zupra.graphics.texture.Texture.initBuffer(@embedFile("assets/bricks/Bricks097_1K-JPG_NormalGL.jpg")) catch unreachable;

    wall_model = Model.fromMesh(gpa, MeshBuilder.cube(12, 3, 1), .{
        .base_color_map = bricks,
        .normal_map = bricks_normal,
        .shading = .pbr,
        .roughness = 0.9,
        .metallic = 0.0,
        .uv_scale = .{ 6, 2 },
    }) catch unreachable;

    // Floor: large rough dielectric plane.
    floor_model = Model.fromMesh(gpa, MeshBuilder.plane(24, 24), .{
        .base_color = .{ .r = 1, .g = 1, .b = 1, .a = 1 },
    }) catch unreachable;

    const albedo = Texture.initBuffer(@embedFile("assets/metal/Metal049A_1K-JPG_Color.jpg")) catch unreachable;
    const normal = Texture.initBuffer(@embedFile("assets/metal/Metal049A_1K-JPG_NormalGL.jpg")) catch unreachable;
    const mr = zupra.graphics.texture.packMetallicRoughness(
        gpa,
        @embedFile("assets/metal/Metal049A_1K-JPG_Roughness.jpg"),
        @embedFile("assets/metal/Metal049A_1K-JPG_Metalness.jpg"),
    ) catch unreachable;

    metal_model = Model.fromMesh(gpa, MeshBuilder.sphere(gpa, 1.0, 48, 48) catch unreachable, .{
        .base_color = .{ .r = 1, .g = 1, .b = 1, .a = 1 },
        .metallic = 1.0, // factor × map.b
        .roughness = 1.0, // factor × map.g (use 1.0 so the map drives it)
        .base_color_map = albedo,
        .normal_map = normal,
        .metallic_roughness_map = mr,
        // normal_flip_y = false,  // GL convention — default; flip to true if bumps invert
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

    helmet_model = zupra.render.gltf.loadMemory(gpa, @embedFile("assets/models/ChronographWatch.glb")) catch unreachable;
    var strength: f32 = 0;
    for (helmet_model.materials) |mat| {
        if (mat.occlusion_strength > strength) strength = mat.occlusion_strength;
    }
    std.debug.print("HIGHEST OCCLUSION STRENGTH: {d}", .{strength});
    // for (helmet_model.materials) |*mat| {
    //     mat.roughness = 1;
    //     mat.metallic = 0;
    // }

    // Lighting: a warm sun + a colored point light.
    env = .{ .ambient = .{ .r = 0.03, .g = 0.03, .b = 0.04, .a = 1 } };
    env.addLight(Light.directional(.{ .x = -0.5, .y = -1.0, .z = -0.35 }, .{ .r = 1.0, .g = 0.96, .b = 0.9, .a = 1 }, 2.5));

    last_ticks = sokol.time.now();
}

pub fn render() void {
    const now = sokol.time.now();
    t += @floatCast(sokol.time.sec(sokol.time.diff(now, last_ticks)));
    last_ticks = now;

    var floor = floor_model.instance();
    floor.setPosition(.{ .x = 0, .y = -0.7, .z = 0 });

    var metal = metal_model.instance();
    metal.setPosition(.{ .x = -1.8, .y = 0, .z = 0 });

    var rough = rough_model.instance();
    rough.setPosition(.{ .x = 0.2, .y = 0, .z = 0 });

    // Glass sweeps left-right in front of the spheres.
    var glass = glass_model.instance();
    glass.setPosition(.{ .x = @sin(t * 0.6) * 2.5, .y = 0.1, .z = 2.2 });

    var wall = wall_model.instance();
    wall.setPosition(.{ .x = 0, .y = 0, .z = -6 });

    var helmet = helmet_model.instance();
    helmet.setPosition(.{ .x = 0, .y = 3, .z = 6 });

    controller.update(zupra.app.getDelta());
    controller.applyTo(&cam);
    scene.begin(cam, env);
    scene.draw(floor);
    scene.draw(metal);
    scene.draw(rough);
    scene.draw(glass);
    scene.draw(wall);
    scene.draw(helmet);
    scene.end();
}

pub fn onEvent(event: *const zupra.Event) void {
    controller.handleEvent(event);
}

pub fn deinit() void {
    floor_model.deinit();
    metal_model.deinit();
    rough_model.deinit();
    glass_model.deinit();
    wall_model.deinit();
    helmet_model.deinit();
    scene.deinit();
    cache.deinit();
}
