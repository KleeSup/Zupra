//! examples/playground.zig
//!
//! BLOOM demo — a night courtyard lit by emissive lanterns.
//!
//! Three deliberate choices, for the same reason the AO rig was matte and dim:
//! an effect is only legible in the conditions that exercise it.
//!
//!   * DARK SCENE. Bloom is a contrast phenomenon. Against a bright sky a glow
//!     has nothing to spread into, so the whole scene sits at low ambient with
//!     no sun and the lanterns are the only real light.
//!   * EMISSIVE FAR ABOVE 1.0. `emissive_strength` is what carries a surface
//!     past the threshold. At 1.0 an emissive material is merely white; the
//!     interesting range starts around 4 and the brightest lantern here is 40.
//!   * A LIGHT PER LANTERN. Emissive geometry does NOT illuminate anything —
//!     there is no global illumination, so an emissive cube is a bright surface
//!     and nothing more. Every engine pairs the two: the emissive makes the
//!     object look like a source, a point light at the same position makes it
//!     act like one. Key 4 separates them so the difference is visible.
//!
//! The lanterns are graded 4 / 10 / 40 left to right. Watch the glow widen with
//! brightness rather than just intensify — that is the thing HDR bloom does that
//! a threshold-and-blur on an LDR image cannot.
//!
//! Keys:
//!   1  bloom on/off      2/3  intensity -/+     5/6  threshold -/+
//!   4  point lights on/off (emissive alone)     7/8  mip count -/+

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

var gpa: std.mem.Allocator = undefined;

var cache: zupra.graphics.pipeline.PipelineCache = undefined;
var scene: zupra.render.SceneRenderer = undefined;
var cam: zupra.render.Camera3D = undefined;
var env: Environment = undefined;
var controller: zupra.render.FirstPersonController = undefined;

var font: zupra.render.Font = undefined;
var batch: zupra.render.SpriteBatch = undefined;
var ui_cam: Camera2D = undefined;
var text: zupra.render.TextBatch2D = undefined;

/// One lantern: an emissive cube, a matching point light, and a post.
const Lantern = struct {
    x: f32,
    z: f32,
    color: zupra.Color,
    /// emissive_strength. The three tiers are the whole point of the scene.
    strength: f32,
    light: LightHandle = undefined,
};

var lanterns = [_]Lantern{
    .{ .x = -7.0, .z = 0.0, .color = .{ .r = 1.0, .g = 0.55, .b = 0.2, .a = 1 }, .strength = 4.0 },
    .{ .x = 0.0, .z = 0.0, .color = .{ .r = 0.4, .g = 0.9, .b = 1.0, .a = 1 }, .strength = 10.0 },
    .{ .x = 7.0, .z = 0.0, .color = .{ .r = 1.0, .g = 0.95, .b = 0.85, .a = 1 }, .strength = 40.0 },
};

const lantern_height: f32 = 2.6;

var floor_model: Model = undefined;
var wall_model: Model = undefined;
var post_model: Model = undefined;
var pillar_model: Model = undefined;
var ball_model: Model = undefined;
/// One emissive cube model per lantern: emissive is a material factor, so each
/// colour and strength needs its own material.
var lamp_models: [lanterns.len]Model = undefined;

var lights_on = true;

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
    scene = zupra.render.SceneRenderer.init(gpa, &cache, .deferred, 1280, 720);
    scene.setAAMethod(.fxaa);

    cam = zupra.render.Camera3D.init(16.0 / 9.0);
    controller = .init(.{ .x = 0, .y = 3.0, .z = -14 });

    font = zupra.render.Font.initFromMemory(gpa, @embedFile("assets/OpenSans-Regular.ttf"), .{}) catch unreachable;
    batch = zupra.render.SpriteBatch.init(gpa, &cache, .{}) catch unreachable;
    ui_cam = Camera2D.init(1280, 720);
    text = zupra.render.TextBatch2D.init(&font, &batch);

    env = Environment.init(gpa);
    // Near-black ambient. Bloom is a contrast effect: raise this and the glow
    // stops registering long before the lanterns stop being bright.
    env.ambient = .{ .r = 0.012, .g = 0.014, .b = 0.020, .a = 1 };
    // No sun at all. Everything visible here is lit by the lanterns.

    const stone = Material{
        .base_color = .{ .r = 0.55, .g = 0.54, .b = 0.56, .a = 1 },
        .metallic = 0.0,
        .roughness = 0.85,
    };
    const metal = Material{
        .base_color = .{ .r = 0.80, .g = 0.80, .b = 0.82, .a = 1 },
        .metallic = 1.0,
        .roughness = 0.22,
    };

    floor_model = Model.fromMesh(gpa, MeshBuilder.plane(60, 60), stone) catch unreachable;
    wall_model = Model.fromMesh(gpa, MeshBuilder.cube(30, 6, 0.5), stone) catch unreachable;
    post_model = Model.fromMesh(gpa, MeshBuilder.cube(0.22, 2.4, 0.22), stone) catch unreachable;
    pillar_model = Model.fromMesh(gpa, MeshBuilder.cube(0.8, 5.0, 0.8), stone) catch unreachable;
    // Polished spheres between the lanterns: a specular highlight of a bright
    // source is itself well above 1.0, so these bloom without being emissive.
    ball_model = Model.fromMesh(gpa, MeshBuilder.sphere(gpa, 0.9, 32, 32) catch unreachable, metal) catch unreachable;

    for (&lanterns, 0..) |*l, i| {
        lamp_models[i] = Model.fromMesh(gpa, MeshBuilder.cube(0.55, 0.55, 0.55), .{
            // Dark base colour on purpose. The glow should come from the
            // emissive term, not from a white surface catching light — with a
            // white base you cannot tell the two apart.
            .base_color = .{ .r = 0.05, .g = 0.05, .b = 0.05, .a = 1 },
            .metallic = 0.0,
            .roughness = 0.4,
            .emissive = l.color,
            .emissive_strength = l.strength,
        }) catch unreachable;

        // The light that actually illuminates. Intensity tracks the emissive
        // strength so the two read as one object; range is deliberately modest
        // so the pools of light stay separate.
        l.light = env.addLight(Light.point(
            .{ .x = l.x, .y = lantern_height, .z = l.z },
            l.color,
            l.strength * 2.5,
            9.0,
        )) catch unreachable;
        if (env.getLight(l.light)) |light| {
            light.shadow = .{
                .enabled = true,
                .resolution = 512,
                .cascade_count = 1,
                .max_distance = 40,
                // .slope_bias = 0,
                // .depth_bias = 0,
            };
        }
    }
}

fn place(model: *Model, x: f32, y: f32, z: f32) void {
    var inst = model.instance();
    inst.setPosition(.{ .x = x, .y = y, .z = z });
    scene.draw(inst);
}

fn placeRotated(model: *Model, x: f32, y: f32, z: f32, angle: f32) void {
    var inst = model.instance();
    inst.setPosition(.{ .x = x, .y = y, .z = z });
    inst.setRotationAxisAngle(.{ .x = 0, .y = 1, .z = 0 }, angle);
    scene.draw(inst);
}

pub fn render() void {
    const dt = zupra.app.getDelta();

    if (zupra.input.isKeyJustPressed(._1)) scene.bloom_enabled = !scene.bloom_enabled;
    if (zupra.input.isKeyJustPressed(._2)) scene.bloom.settings.intensity = @max(0.0, scene.bloom.settings.intensity - 0.02);
    if (zupra.input.isKeyJustPressed(._3)) scene.bloom.settings.intensity = @min(1.0, scene.bloom.settings.intensity + 0.02);
    if (zupra.input.isKeyJustPressed(._5)) scene.bloom.settings.threshold = @max(0.0, scene.bloom.settings.threshold - 0.25);
    if (zupra.input.isKeyJustPressed(._6)) scene.bloom.settings.threshold = @min(10.0, scene.bloom.settings.threshold + 0.25);

    // Emissive geometry lights nothing by itself. Turning the point lights off
    // leaves three glowing cubes floating in a black courtyard, which is exactly
    // what emissive-without-a-light looks like and why engines pair them.
    if (zupra.input.isKeyJustPressed(._4)) {
        lights_on = !lights_on;
        for (lanterns) |l| {
            if (env.getLight(l.light)) |light| {
                light.intensity = if (lights_on) l.strength * 2.5 else 0.0;
            }
        }
    }

    // Mip count changes the chain length, so the targets have to be rebuilt.
    if (zupra.input.isKeyJustPressed(._7) and scene.bloom.settings.mip_count > 1) {
        scene.bloom.settings.mip_count -= 1;
        scene.bloom.rebuild();
    }
    if (zupra.input.isKeyJustPressed(._8) and scene.bloom.settings.mip_count < zupra.render.bloom_max_mips) {
        scene.bloom.settings.mip_count += 1;
        scene.bloom.rebuild();
    }

    controller.update(dt);
    controller.applyTo(&cam);

    scene.begin(cam, &env);

    place(&floor_model, 0, 0, 0);
    placeRotated(&wall_model, 0, 3, 10.0, 0);
    placeRotated(&wall_model, -15.0, 3, 0, std.math.pi * 0.5);
    placeRotated(&wall_model, 15.0, 3, 0, std.math.pi * 0.5);

    // Pillars set back from the lanterns, so each casts a long shadow across the
    // floor. Bloom around a source with a hard shadow edge nearby is the honest
    // test that the glow is additive light and not a blur of the whole image.
    place(&pillar_model, -10.5, 2.5, 5.0);
    place(&pillar_model, -3.5, 2.5, 5.0);
    place(&pillar_model, 3.5, 2.5, 5.0);
    place(&pillar_model, 10.5, 2.5, 5.0);

    for (lanterns, 0..) |l, i| {
        place(&post_model, l.x, 1.2, l.z);

        // The point light sits INSIDE this cube. Left casting, the cube shadows
        // its own light in all six cube-map directions and the lantern's entire
        // range goes black — which is exactly what a light fixture does if you
        // let it cast.
        var lamp = lamp_models[i].instance();
        lamp.setPosition(.{ .x = l.x, .y = lantern_height, .z = l.z });
        lamp.cast_shadows = false;
        scene.draw(lamp);
    }

    // Metal spheres between the lanterns: their specular highlights are far
    // above 1.0, so they bloom without carrying any emissive of their own.
    place(&ball_model, -3.5, 0.9, -2.0);
    place(&ball_model, 3.5, 0.9, -2.0);
    place(&ball_model, 0.0, 0.9, -5.0);

    scene.end();

    fps_accum += dt;
    fps_frames += 1;
    if (fps_accum >= 0.25) {
        const fps = @as(f32, @floatFromInt(fps_frames)) / fps_accum;
        // const b = scene.bloom.settings;
        // fps_text = std.fmt.bufPrint(
        //     &fps_buf,
        //     "{d:.0} FPS {d:.2} ms | bloom {s}  intensity {d:.2}  threshold {d:.2}  mips {d} | lights {s}  [1 bloom  2/3 intensity  5/6 threshold  4 lights  7/8 mips]",
        //     .{
        //         fps,                                      1000.0 / fps,
        //         if (scene.bloom_enabled) "ON" else "OFF", b.intensity,
        //         b.threshold,                              scene.bloom.mip_count,
        //         if (lights_on) "on" else "off",
        //     },
        // ) catch "FPS ?";
        const cs = env.lighting.clusterStats();
        fps_text = std.fmt.bufPrint(
            &fps_buf,
            "{d:.0} FPS {d:.2} ms | froxels max {d} avg {d:.1} ({d}/{d} used, {d} full, {d} dropped)",
            .{
                fps,           1000.0 / fps,
                cs.max_lights, cs.avg_lights,
                cs.occupied,   cs.total,
                cs.saturated,  cs.dropped,
            },
        ) catch "FPS ?";
        fps_accum = 0;
        fps_frames = 0;
    }

    ui_cam.setViewport(sokol.app.widthf(), sokol.app.heightf());
    zupra.beginDrawing();
    text.begin(ui_cam);
    text.draw(fps_text, 12, 12, 0.34, zupra.colors.WHITE);
    text.end();
    zupra.endDrawing();
}

pub fn onEvent(event: *const zupra.Event) void {
    controller.handleEvent(event);
}

pub fn deinit() void {
    font.deinit();
    batch.deinit(gpa);
    floor_model.deinit();
    wall_model.deinit();
    post_model.deinit();
    pillar_model.deinit();
    ball_model.deinit();
    for (&lamp_models) |*m| m.deinit();
    env.deinit();
    scene.deinit();
    cache.deinit();
}
