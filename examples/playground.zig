//! examples/playground.zig
//!
//! AMBIENT OCCLUSION demo.
//!
//! Built to make SSAO legible rather than to stress the renderer, which means
//! three deliberate choices:
//!
//!   * MATTE, LIGHT, NON-METALLIC materials. Occlusion scales the ambient
//!     DIFFUSE term. A metal has almost no diffuse response, and a dark albedo
//!     has little to darken, so either one hides the effect completely — it
//!     looks like a broken feature when it is working correctly.
//!   * AMBIENT-DOMINANT LIGHTING. A strong sun swamps ambient, and AO does not
//!     touch direct light at all (that is what shadow maps are for). The sun
//!     here is deliberately weak, and key 7 removes it entirely.
//!   * CONTACT-HEAVY GEOMETRY. Occlusion lives where surfaces meet: inside
//!     corners, under objects resting on the floor, in the gap between two
//!     boxes pushed together. A field of well-spaced spheres shows almost none.
//!
//! Deferred only — SSAO reads the G-buffer's normals, which the forward path
//! does not produce.
//!
//! Keys:
//!   1  SSAO on/off        2/3  radius -/+       4/5  intensity -/+
//!   6  half-res on/off    7    sun on/off       8    ambient -/+ (cycles)

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
const Sprite = zupra.graphics.texture.Sprite;
const TextureRegion = zupra.graphics.texture.TextureRegion;

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

// Every surface is the same matte off-white. Uniform material is the point:
// with nothing else varying, any shading difference on screen is geometry-driven
// occlusion rather than a material property.
const chalk = Material{
    .base_color = .{ .r = 0.82, .g = 0.80, .b = 0.78, .a = 1 },
    .metallic = 0.0,
    .roughness = 0.95,
};

var floor_model: Model = undefined;
var wall_model: Model = undefined;
var block_model: Model = undefined;
var small_model: Model = undefined;
var pillar_model: Model = undefined;
var ball_model: Model = undefined;

var sun: LightHandle = undefined;
var sun_on = true;

const ambient_steps = [_]f32{ 0.10, 0.20, 0.35, 0.50 };
var ambient_index: usize = 2;

/// Draw the raw occlusion buffer over the scene. The most direct question you
/// can ask the pass -- white where open, dark in creases. An all-black or
/// all-white buffer is instantly recognisable here and nearly impossible to
/// diagnose from the lit image alone.
var show_ao_buffer = false;

var fps_accum: f32 = 0;
var fps_frames: u32 = 0;
var fps_buf: [256]u8 = undefined;
var fps_text: []const u8 = "";

pub fn main(ctx: std.process.Init) !void {
    zupra.init(ctx, .{ .initFn = init, .renderFn = render, .deinitFn = deinit, .eventFn = onEvent });
}

fn applyAmbient() void {
    const a = ambient_steps[ambient_index];
    env.ambient = .{ .r = a, .g = a * 0.99, .b = a * 0.97, .a = 1 };
}

pub fn init() void {
    gpa = zupra.getGPA();
    cache = .init(gpa);

    // Deferred: SSAO needs G-buffer normals.
    scene = zupra.render.SceneRenderer.init(gpa, &cache, .deferred, 1280, 720);
    scene.setAAMethod(.fxaa);

    cam = zupra.render.Camera3D.init(16.0 / 9.0);
    // Low and close. AO is a contact-scale effect; from a distance the contact
    // shadows are a few pixels wide and invisible.
    controller = .init(.{ .x = 6, .y = 2.4, .z = -9 });

    font = zupra.render.Font.initFromMemory(gpa, @embedFile("assets/OpenSans-Regular.ttf"), .{}) catch unreachable;
    batch = zupra.render.SpriteBatch.init(gpa, &cache, .{}) catch unreachable;
    ui_cam = Camera2D.init(1280, 720);
    text = zupra.render.TextBatch2D.init(&font, &batch);

    env = Environment.init(gpa);
    applyAmbient();

    // Weak sun, angled so the corner walls stay partly self-shaded. Its job is
    // to give the scene some form, not to light it — turn it off with 7 and the
    // remaining image is pure ambient, which is where AO does all its work.
    sun = env.addLight(Light.directional(
        .{ .x = -0.45, .y = -1.0, .z = -0.30 },
        .{ .r = 1.0, .g = 0.97, .b = 0.92, .a = 1 },
        0.8,
    )) catch unreachable;
    if (env.getLight(sun)) |l| {
        l.shadow = .{ .enabled = true, .resolution = 1024, .cascade_count = 3, .max_distance = 45 };
    }

    floor_model = Model.fromMesh(gpa, MeshBuilder.plane(40, 40), chalk) catch unreachable;
    wall_model = Model.fromMesh(gpa, MeshBuilder.cube(14, 5, 0.4), chalk) catch unreachable;
    block_model = Model.fromMesh(gpa, MeshBuilder.cube(2, 2, 2), chalk) catch unreachable;
    small_model = Model.fromMesh(gpa, MeshBuilder.cube(0.9, 0.9, 0.9), chalk) catch unreachable;
    pillar_model = Model.fromMesh(gpa, MeshBuilder.cube(0.7, 4.5, 0.7), chalk) catch unreachable;
    ball_model = Model.fromMesh(gpa, MeshBuilder.sphere(gpa, 0.85, 32, 32) catch unreachable, chalk) catch unreachable;
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

    if (zupra.input.isKeyJustPressed(._1)) scene.ssao_enabled = !scene.ssao_enabled;
    if (zupra.input.isKeyJustPressed(._2)) scene.ssao.settings.radius = @max(0.05, scene.ssao.settings.radius - 0.1);
    if (zupra.input.isKeyJustPressed(._3)) scene.ssao.settings.radius = @min(4.0, scene.ssao.settings.radius + 0.1);
    if (zupra.input.isKeyJustPressed(._4)) scene.ssao.settings.intensity = @max(0.0, scene.ssao.settings.intensity - 0.1);
    if (zupra.input.isKeyJustPressed(._5)) scene.ssao.settings.intensity = @min(3.0, scene.ssao.settings.intensity + 0.1);
    if (zupra.input.isKeyJustPressed(._6)) {
        scene.ssao.settings.half_resolution = !scene.ssao.settings.half_resolution;
        // rebuild, not resize: the window size hasn't changed, so resize would
        // early-out and leave the targets at the old scale.
        scene.ssao.rebuild(scene.width, scene.height);
    }
    if (zupra.input.isKeyJustPressed(._9)) show_ao_buffer = !show_ao_buffer;
    if (zupra.input.isKeyJustPressed(._7)) {
        sun_on = !sun_on;
        if (env.getLight(sun)) |l| l.intensity = if (sun_on) 0.8 else 0.0;
    }
    if (zupra.input.isKeyJustPressed(._8)) {
        ambient_index = (ambient_index + 1) % ambient_steps.len;
        applyAmbient();
    }

    controller.update(dt);
    controller.applyTo(&cam);

    scene.begin(cam, &env);

    place(&floor_model, 0, 0, 0);

    // An inside corner. Two walls meeting is the textbook AO case: the seam
    // where they join, and both seams where they meet the floor, should darken
    // gradually rather than reading as three flat planes butted together.
    placeRotated(&wall_model, 0, 2.5, 7.0, 0);
    placeRotated(&wall_model, -7.0, 2.5, 0, std.math.pi * 0.5);

    // Blocks pushed together with narrow gaps. The gaps matter more than the
    // blocks: a 0.3-unit slot between two boxes is exactly the scale `radius`
    // controls, so widening the radius past ~1.0 washes these out.
    place(&block_model, -4.0, 1.0, 3.0);
    place(&block_model, -1.7, 1.0, 3.0);
    place(&block_model, -2.85, 3.0, 3.0); // stacked across the seam below

    // Small blocks nestled into the corner between a big block and the floor.
    place(&small_model, 0.9, 0.45, 3.0);
    place(&small_model, 1.85, 0.45, 3.0);
    place(&small_model, 1.4, 1.35, 3.0);

    // Pillars, spaced to close up as you walk past — the gap between them
    // darkens progressively, which shows the falloff is smooth rather than a
    // hard threshold.
    var i: usize = 0;
    while (i < 4) : (i += 1) {
        const x = -5.0 + @as(f32, @floatFromInt(i)) * 1.1;
        place(&pillar_model, x, 2.25, 5.6);
    }

    // Balls resting ON the floor, not floating above it. The contact ring under
    // each is the single clearest read on whether AO is working: without it
    // they look pasted onto the surface.
    place(&ball_model, 2.5, 0.85, 0.5);
    place(&ball_model, 4.2, 0.85, 1.2);
    place(&ball_model, 3.3, 0.85, 2.4); // three in a cluster, mutually occluding

    // One ball tucked against a block, so occlusion appears on the vertical face
    // too and not only on the ground.
    place(&block_model, -0.5, 1.0, -2.0);
    place(&ball_model, 1.0, 0.85, -2.0);

    scene.end();

    fps_accum += dt;
    fps_frames += 1;
    if (fps_accum >= 0.25) {
        const fps = @as(f32, @floatFromInt(fps_frames)) / fps_accum;
        const s = scene.ssao.settings;
        fps_text = std.fmt.bufPrint(
            &fps_buf,
            "{d:.0} FPS {d:.2} ms | SSAO {s} r={d:.2} i={d:.2} {s} | sun {s} | ambient {d:.2}  [1 ssao  2/3 radius  4/5 intensity  6 res  7 sun  8 ambient  9 view AO]",
            .{
                fps,                                     1000.0 / fps,
                if (scene.ssao_enabled) "ON" else "OFF", s.radius,
                s.intensity,                             if (s.half_resolution) "half" else "full",
                if (sun_on) "on" else "off",             ambient_steps[ambient_index],
            },
        ) catch "FPS ?";
        fps_accum = 0;
        fps_frames = 0;
    }

    ui_cam.setViewport(sokol.app.widthf(), sokol.app.heightf());
    zupra.beginDrawing();

    if (show_ao_buffer) {
        // Stretch the AO target over the whole window. It's R8, so the sprite
        // shader shows occlusion in the red channel -- open areas read bright,
        // creases dark. Colour is irrelevant here; the spatial pattern is the
        // whole point.
        var ao_sprite = Sprite.init(TextureRegion.full(scene.ssao.debugTexture()));
        ao_sprite.dest = .{
            .x = 0,
            .y = 0,
            .width = sokol.app.widthf(),
            .height = sokol.app.heightf(),
        };
        batch.begin(ui_cam, .none);
        batch.draw(ao_sprite);
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
    font.deinit();
    batch.deinit(gpa);
    floor_model.deinit();
    wall_model.deinit();
    block_model.deinit();
    small_model.deinit();
    pillar_model.deinit();
    ball_model.deinit();
    env.deinit();
    scene.deinit();
    cache.deinit();
}
