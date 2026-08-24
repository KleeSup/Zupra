//! examples/playground.zig
//!
//! RenderWorld and motion vector test.
//!
//! Built to make ghosting obvious when it is present and clearly absent when it
//! is fixed. The arrangement is deliberate in three ways.
//!
//! Motion is fast and lateral. A slow or head-on movement produces a small
//! screen-space offset, and camera reprojection alone is nearly right for it, so
//! a broken velocity buffer would look almost correct. Objects here cross the
//! screen quickly, which is where the difference is largest.
//!
//! Moving objects pass in front of high contrast backgrounds. Ghosting is a
//! stale colour blended into the current one, so it only shows where the two
//! differ. A pale object over a pale floor hides it almost completely, and the
//! dark banded wall behind the track exists for exactly this reason.
//!
//! Every kind of motion is represented. Translation alone would not catch a
//! velocity pass that ignores rotation, since per-object velocity and per-vertex
//! velocity agree for pure translation and disagree everywhere else. So there
//! are spinning objects, orbiting objects, and one that scales.
//!
//! The moving lamp carries a small emissive marker so the light source is
//! visible rather than inferred, and moves independently of the geometry it
//! lights, which puts moving shadows across static objects.
//!
//! Keys:
//!   1  TAA on/off        2  pause all motion       3  velocity buffer view
//!   4  slow motion       5  bloom on/off           6  AO on/off

const std = @import("std");
const zupra = @import("zupra");
const sokol = zupra.intern.sokol;

const Model = zupra.render.Model;
const Material = zupra.render.Material;
const MeshBuilder = zupra.render.MeshBuilder;
const Light = zupra.render.Light;
const LightHandle = zupra.render.LightHandle;
const Environment = zupra.render.Environment;
const Camera2D = zupra.render.Camera2D;
const RenderWorld = zupra.render.RenderWorld;
const RenderHandle = zupra.render.RenderHandle;
const Sprite = zupra.graphics.texture.Sprite;
const TextureRegion = zupra.graphics.texture.TextureRegion;

var gpa: std.mem.Allocator = undefined;

var cache: zupra.graphics.pipeline.PipelineCache = undefined;
var scene: zupra.render.SceneRenderer = undefined;
var cam: zupra.render.Camera3D = undefined;
var env: Environment = undefined;
var controller: zupra.render.FirstPersonController = undefined;
var world: RenderWorld = undefined;

var font: zupra.render.Font = undefined;
var batch: zupra.render.SpriteBatch = undefined;
var ui_cam: Camera2D = undefined;
var text: zupra.render.TextBatch2D = undefined;

var floor_model: Model = undefined;
var backdrop_light: Model = undefined;
var backdrop_dark: Model = undefined;
var pillar_model: Model = undefined;
var runner_model: Model = undefined;
var spinner_model: Model = undefined;
var pulser_model: Model = undefined;
var lamp_marker: Model = undefined;
var lamp_post: Model = undefined;
var truck: Model = undefined;

/// Objects that translate across the screen. Lateral motion at speed is where a
/// missing velocity buffer shows up most clearly.
const Runner = struct {
    handle: RenderHandle = undefined,
    z: f32,
    speed: f32,
    phase: f32,
    height: f32,
};
var runners = [_]Runner{
    .{ .z = -2.0, .speed = 7.0, .phase = 0.0, .height = 0.8 },
    .{ .z = 1.0, .speed = -5.5, .phase = 1.7, .height = 1.6 },
    .{ .z = 4.0, .speed = 9.0, .phase = 3.1, .height = 2.6 },
};

/// Objects that rotate in place. Rotation is the case per-object velocity gets
/// wrong and per-vertex velocity gets right, since parts of the object move at
/// different rates.
const Spinner = struct {
    handle: RenderHandle = undefined,
    x: f32,
    speed: f32,
};
var spinners = [_]Spinner{
    .{ .x = -6.0, .speed = 2.2 },
    .{ .x = 0.0, .speed = -3.4 },
    .{ .x = 6.0, .speed = 1.6 },
};

/// One object that scales, since a changing scale is a third distinct case.
var pulser: RenderHandle = undefined;

var truck_handle: RenderHandle = undefined;

var lamp_marker_handle: RenderHandle = undefined;
var lamp_post_handle: RenderHandle = undefined;
var lamp_light: LightHandle = undefined;

var elapsed: f32 = 0;
var paused = false;
var slow = false;
var show_velocity = false;
var render_ao = false;

var fps_accum: f32 = 0;
var fps_frames: u32 = 0;
var fps_buf: [256]u8 = undefined;
var fps_text: []const u8 = "";

pub fn main(ctx: std.process.Init) !void {
    zupra.init(ctx, .{ .initFn = init, .renderFn = render, .deinitFn = deinit, .eventFn = onEvent });
}

pub fn init() void {
    gpa = zupra.getGPA();
    cache = .init(gpa);
    scene = zupra.render.SceneRenderer.init(gpa, &cache, .deferred, 1280, 720) catch unreachable;
    scene.setAAMethod(.taa);

    cam = zupra.render.Camera3D.init(16.0 / 9.0);
    // Side on to the track, so the runners cross the view rather than approach
    // it. Screen-space motion is largest here, and so is any error in it.
    controller = .init(.{ .x = 0, .y = 4.0, .z = -18 });

    font = zupra.render.Font.initFromMemory(gpa, @embedFile("assets/OpenSans-Regular.ttf"), .{}) catch unreachable;
    batch = zupra.render.SpriteBatch.init(gpa, &cache, .{}) catch unreachable;
    ui_cam = Camera2D.init(1280, 720);
    text = zupra.render.TextBatch2D.init(&font, &batch);

    env = Environment.init(gpa);
    env.ambient = .{ .r = 0.10, .g = 0.10, .b = 0.12, .a = 1 };

    _ = env.addLight(Light.directional(
        .{ .x = -0.4, .y = -1.0, .z = -0.3 },
        .{ .r = 0.85, .g = 0.88, .b = 1.0, .a = 1 },
        1.2,
    )) catch unreachable;

    // The travelling lamp. Moves independently of the geometry it lights, so
    // shadows sweep across static objects and moving objects alike.
    lamp_light = env.addLight(Light.point(
        .{ .x = 0, .y = 5.0, .z = 0 },
        .{ .r = 1.0, .g = 0.85, .b = 0.55, .a = 1 },
        30.0,
        18.0,
    )) catch unreachable;
    if (env.getLight(lamp_light)) |l| {
        l.shadow = .{ .enabled = true, .resolution = 512, .cascade_count = 1, .max_distance = 60 };
    }

    const pale = Material{
        .base_color = .{ .r = 0.82, .g = 0.82, .b = 0.84, .a = 1 },
        .metallic = 0.0,
        .roughness = 0.85,
    };
    const dark = Material{
        .base_color = .{ .r = 0.09, .g = 0.09, .b = 0.11, .a = 1 },
        .metallic = 0.0,
        .roughness = 0.9,
    };
    const bright = Material{
        .base_color = .{ .r = 0.95, .g = 0.93, .b = 0.88, .a = 1 },
        .metallic = 0.0,
        .roughness = 0.5,
    };

    floor_model = Model.fromMesh(gpa, MeshBuilder.plane(60, 60), pale) catch unreachable;
    // Alternating light and dark panels behind the track. Ghosting is a stale
    // colour mixed into the current one, so it is only visible where the two
    // differ, and a moving object needs something to contrast against.
    backdrop_light = Model.fromMesh(gpa, MeshBuilder.cube(4, 8, 0.4), bright) catch unreachable;
    backdrop_dark = Model.fromMesh(gpa, MeshBuilder.cube(4, 8, 0.4), dark) catch unreachable;
    pillar_model = Model.fromMesh(gpa, MeshBuilder.cube(0.9, 5.0, 0.9), pale) catch unreachable;
    runner_model = Model.fromMesh(gpa, MeshBuilder.sphere(gpa, 0.7, 32, 32) catch unreachable, bright) catch unreachable;
    spinner_model = Model.fromMesh(gpa, MeshBuilder.cube(1.6, 1.6, 1.6), bright) catch unreachable;
    pulser_model = Model.fromMesh(gpa, MeshBuilder.sphere(gpa, 1.0, 32, 32) catch unreachable, bright) catch unreachable;
    lamp_marker = Model.fromMesh(gpa, MeshBuilder.cube(0.5, 0.5, 0.5), .{
        .base_color = .{ .r = 0.05, .g = 0.05, .b = 0.05, .a = 1 },
        .emissive = .{ .r = 1.0, .g = 0.85, .b = 0.55, .a = 1 },
        .emissive_strength = 12.0,
    }) catch unreachable;
    lamp_post = Model.fromMesh(gpa, MeshBuilder.cube(0.12, 1.2, 0.12), pale) catch unreachable;

    truck = zupra.render.gltf.loadMemory(gpa, @embedFile("assets/Icecream truck Texturing Versuch 1.glb")) catch unreachable;
    //truck.setShadingModel(.lambert);

    //lamp_marker.setShadingModel(.lambert);

    world = RenderWorld.init(gpa);

    truck_handle = world.add(.{
        .model = &truck,
        .position = .{ .x = 0, .y = 0, .z = -3 },
        .mobility = .static,
        //.cast_shadows = false,
    }) catch unreachable;

    // Static geometry. Marked .static so it is never refitted and, once shadow
    // caching lands, never redrawn into the atlas either.
    _ = world.add(.{ .model = &floor_model, .mobility = .static }) catch unreachable;

    var i: usize = 0;
    while (i < 9) : (i += 1) {
        const x = -16.0 + @as(f32, @floatFromInt(i)) * 4.0;
        const model = if (i % 2 == 0) &backdrop_light else &backdrop_dark;
        _ = world.add(.{
            .model = model,
            .position = .{ .x = x, .y = 4.0, .z = 8.0 },
            .mobility = .static,
        }) catch unreachable;
    }

    i = 0;
    while (i < 5) : (i += 1) {
        const x = -12.0 + @as(f32, @floatFromInt(i)) * 6.0;
        _ = world.add(.{
            .model = &pillar_model,
            .position = .{ .x = x, .y = 2.5, .z = 2.0 },
            .mobility = .static,
        }) catch unreachable;
    }

    for (&runners) |*r| {
        r.handle = world.add(.{
            .model = &runner_model,
            .position = .{ .x = 0, .y = r.height, .z = r.z },
            .mobility = .dynamic,
        }) catch unreachable;
    }

    for (&spinners) |*s| {
        s.handle = world.add(.{
            .model = &spinner_model,
            .position = .{ .x = s.x, .y = 1.0, .z = -6.0 },
            .mobility = .dynamic,
        }) catch unreachable;
    }

    pulser = world.add(.{
        .model = &pulser_model,
        .position = .{ .x = 10.0, .y = 1.6, .z = -4.0 },
        .mobility = .dynamic,
    }) catch unreachable;

    // The lamp marker does not cast, since it surrounds its own light source.
    lamp_marker_handle = world.add(.{
        .model = &lamp_marker,
        .position = .{ .x = 0, .y = 5.0, .z = 0 },
        .mobility = .dynamic,
        .cast_shadows = false,
    }) catch unreachable;
    lamp_post_handle = world.add(.{
        .model = &lamp_post,
        .position = .{ .x = 0, .y = 4.2, .z = 0 },
        .mobility = .dynamic,
    }) catch unreachable;
}

fn axisAngle(axis: zupra.math.Vec3, angle: f32) zupra.math.Quaternion {
    return zupra.math.zm.quatFromAxisAngle(
        zupra.math.zm.f32x4(axis.x, axis.y, axis.z, 0),
        angle,
    );
}

const identity_quat = zupra.math.Quaternion{ 0, 0, 0, 1 };
const unit_scale = zupra.math.Vec3{ .x = 1, .y = 1, .z = 1 };

pub fn render() void {
    const dt = zupra.app.getDelta();
    if (!paused) elapsed += dt * (if (slow) @as(f32, 0.15) else 1.0);

    if (zupra.input.isKeyJustPressed(._1)) {
        scene.setAAMethod(if (scene.isTaaActive()) .none else .taa);
    }
    if (zupra.input.isKeyJustPressed(._2)) paused = !paused;
    if (zupra.input.isKeyJustPressed(._3)) {
        show_velocity = !show_velocity;
        if (show_velocity) {
            scene.setAAMethod(.fxaa_quality);
            //scene.setRenderScale(2);
        } else {
            scene.setAAMethod(.none);
            scene.setRenderScale(1);
        }
    }
    if (zupra.input.isKeyJustPressed(._4)) slow = !slow;
    if (zupra.input.isKeyJustPressed(._5)) scene.bloom_enabled = !scene.bloom_enabled;
    if (zupra.input.isKeyJustPressed(._6)) scene.setSsaoEnabled(!scene.ssao_enabled);
    if (zupra.input.isKeyJustPressed(._9)) render_ao = !render_ao;

    // Translation. Wraps at the ends of the track rather than reversing, so
    // there is no moment of zero velocity to hide behind.
    for (runners) |r| {
        const span: f32 = 34.0;
        const t = @mod(elapsed * r.speed + r.phase * 10.0, span);
        world.setPosition(r.handle, .{ .x = -17.0 + t, .y = r.height, .z = r.z });
    }

    // Rotation, which per-object velocity cannot represent correctly.
    for (spinners) |s| {
        world.setTransform(
            s.handle,
            .{ .x = s.x, .y = 1.0, .z = -6.0 },
            axisAngle(.{ .x = 0.3, .y = 1.0, .z = 0.15 }, elapsed * s.speed),
            unit_scale,
        );
    }

    // Scale, the third case.
    const pulse = 0.6 + 0.5 * (1.0 + @sin(elapsed * 2.0));
    world.setTransform(
        pulser,
        .{ .x = 10.0, .y = 1.6, .z = -4.0 },
        identity_quat,
        .{ .x = pulse, .y = pulse, .z = pulse },
    );

    // The lamp sweeps the track. Its light and its marker share a position, so
    // the source is visible rather than inferred.
    const lamp_x = @sin(elapsed * 0.6) * 13.0;
    const lamp_z = @cos(elapsed * 0.35) * 4.0;
    if (env.getLight(lamp_light)) |l| {
        l.position = .{ .x = lamp_x, .y = 5.0, .z = lamp_z };
    }
    world.setPosition(lamp_marker_handle, .{ .x = lamp_x, .y = 5.0, .z = lamp_z });
    world.setPosition(lamp_post_handle, .{ .x = lamp_x, .y = 4.2, .z = lamp_z });

    controller.update(dt);
    controller.applyTo(&cam);

    scene.begin(cam, &env);
    world.submit(&scene, cam);
    scene.end();

    fps_accum += dt;
    fps_frames += 1;
    if (fps_accum >= 0.25) {
        const fps = @as(f32, @floatFromInt(fps_frames)) / fps_accum;
        const ws = world.stats;
        const cached_pct: f32 = if (scene.shadow_instances > 0)
            100.0 * @as(f32, @floatFromInt(scene.shadow_static_instances)) /
                @as(f32, @floatFromInt(scene.shadow_instances))
        else
            0;
        fps_text = std.fmt.bufPrint(
            &fps_buf,
            "{d:.0} FPS {d:.2}ms | TAA {s} | objects {d}/{d} visible, {d} moved | {s}{s}  [1 taa 2 pause 3 velocity 4 slow 5 bloom 6 ao]\nshadow: {d} draws, {d} instances, {d:.0}% cacheable",
            .{
                fps,                                      1000.0 / fps,
                if (scene.isTaaActive()) "on" else "off", ws.visible,
                ws.total,                                 ws.moved,
                if (paused) "paused " else "",            if (slow) "slow" else "",
                scene.shadow_draws,                       scene.shadow_instances,
                cached_pct,
            },
        ) catch "FPS ?";
        fps_accum = 0;
        fps_frames = 0;
    }

    ui_cam.setViewport(sokol.app.widthf(), sokol.app.heightf());
    zupra.beginDrawing();

    if (render_ao) {
        // Red and green are the signed screen-space offset to where each surface
        // was. Static geometry should be black, since the pass only draws
        // objects that moved.
        var v = Sprite.init(TextureRegion.full(scene.ssao.debugTexture()));
        v.dest = .{ .x = 0, .y = 0, .width = sokol.app.widthf(), .height = sokol.app.heightf() };
        batch.begin(ui_cam, .none);
        batch.draw(v);
        batch.end();
    }

    text.begin(ui_cam);
    text.draw(fps_text, 12, 12, 0.34, zupra.colors.WHITE);
    text.end();
    zupra.endDrawing();
}

pub fn onEvent(event: *const zupra.Event) void {
    controller.handleEvent(event);
}

pub fn deinit() void {
    world.deinit();
    font.deinit();
    batch.deinit(gpa);
    floor_model.deinit();
    backdrop_light.deinit();
    backdrop_dark.deinit();
    pillar_model.deinit();
    runner_model.deinit();
    spinner_model.deinit();
    pulser_model.deinit();
    lamp_marker.deinit();
    lamp_post.deinit();
    truck.deinit();
    env.deinit();
    scene.deinit();
    cache.deinit();
}
