//! examples/playground.zig
//!
//! AO VALIDATION RIG.
//!
//! Built to make the AO buffer CHECKABLE rather than merely viewable. Judging
//! occlusion from a lit screenshot is close to useless: a uniform 10% darkening
//! and a correct result look much the same, which is exactly how a global error
//! survives several rounds of tuning the crease behaviour.
//!
//! So the scene is arranged around regions whose correct answer is known in
//! advance:
//!
//!   * A LARGE EMPTY FLOOR. Nothing within any plausible radius. Correct AO here
//!     is exactly 1.0 -- pure white in the raw buffer, and the lit image
//!     identical whether AO is on or off. If this region darkens, the bug is in
//!     the open-surface path and no amount of crease tuning will find it.
//!   * A SINGLE ISOLATED SPHERE on that floor. The contact ring is the only
//!     thing that should darken. Its silhouette against the empty floor is where
//!     a rim artifact shows up unambiguously, with nothing else nearby to
//!     confuse it.
//!   * AN INSIDE CORNER, far from everything else. Should darken smoothly toward
//!     the seam and nowhere else.
//!   * A STAIR of blocks with 90-degree inside angles at three different scales,
//!     to check that the radius resolves detail at the scale it claims to.
//!
//! Everything is the same matte white: occlusion scales the ambient DIFFUSE
//! term, so metal or dark albedo hides it, and a single material means any
//! shading difference is geometric.
//!
//! Keys:
//!   1  AO on/off        9  view raw AO buffer (the one that matters)
//!   2/3  radius -/+     4/5  intensity -/+     6/7  slices -/+   8  steps cycle
//!   0  bloom + TAA off (isolate AO completely)

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

/// One matte white material everywhere. Occlusion acts on ambient diffuse, so a
/// metal or a dark base colour hides it; uniform material means every shading
/// difference on screen is geometry.
const chalk = Material{
    .base_color = .{ .r = 0.85, .g = 0.85, .b = 0.85, .a = 1 },
    .metallic = 0.0,
    .roughness = 1.0,
};

var floor_model: Model = undefined;
var wall_model: Model = undefined;
var ball_model: Model = undefined;
var block_l: Model = undefined;
var block_m: Model = undefined;
var block_s: Model = undefined;

var show_ao_buffer = false;
var isolate = false;

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
    scene = zupra.render.SceneRenderer.init(gpa, &cache, .forward, 1280, 720);
    scene.setAAMethod(.none); // nothing between the AO and the eye

    cam = zupra.render.Camera3D.init(16.0 / 9.0);
    // Standing on the empty half of the floor, looking toward the test geometry.
    controller = .init(.{ .x = 0, .y = 2.2, .z = -16 });

    font = zupra.render.Font.initFromMemory(gpa, @embedFile("assets/OpenSans-Regular.ttf"), .{}) catch unreachable;
    batch = zupra.render.SpriteBatch.init(gpa, &cache, .{}) catch unreachable;
    ui_cam = Camera2D.init(1280, 720);
    text = zupra.render.TextBatch2D.init(&font, &batch);

    env = Environment.init(gpa);
    // Ambient-dominant. AO only scales the ambient term, so a scene carried by
    // analytic lights would show almost none of it.
    env.ambient = .{ .r = 0.42, .g = 0.42, .b = 0.45, .a = 1 };

    // One weak light purely for shape. AO must be visible without it -- if
    // toggling this changes the AO, something is wired wrong.
    _ = env.addLight(Light.directional(
        .{ .x = -0.4, .y = -1.0, .z = -0.25 },
        .{ .r = 1, .g = 1, .b = 1, .a = 1 },
        0.5,
    )) catch unreachable;

    // scene.ssao.settings = .{
    //     .slices = 3,
    //     .steps = 4,
    //     .radius = 1,
    //     .intensity = 2,
    //     .blur_radius = 2,
    //     .half_resolution = true,
    // };

    floor_model = Model.fromMesh(gpa, MeshBuilder.plane(80, 80), chalk) catch unreachable;
    wall_model = Model.fromMesh(gpa, MeshBuilder.cube(12, 6, 0.4), chalk) catch unreachable;
    ball_model = Model.fromMesh(gpa, MeshBuilder.sphere(gpa, 1.2, 48, 48) catch unreachable, chalk) catch unreachable;
    block_l = Model.fromMesh(gpa, MeshBuilder.cube(3.0, 3.0, 3.0), chalk) catch unreachable;
    block_m = Model.fromMesh(gpa, MeshBuilder.cube(1.0, 1.0, 1.0), chalk) catch unreachable;
    block_s = Model.fromMesh(gpa, MeshBuilder.cube(0.3, 0.3, 0.3), chalk) catch unreachable;
}

fn place(model: *Model, x: f32, y: f32, z: f32) void {
    var inst = model.instance();
    inst.setPosition(.{ .x = x, .y = y, .z = z });
    scene.draw(inst);
}

fn placeRot(model: *Model, x: f32, y: f32, z: f32, a: f32) void {
    var inst = model.instance();
    inst.setPosition(.{ .x = x, .y = y, .z = z });
    inst.setRotationAxisAngle(.{ .x = 0, .y = 1, .z = 0 }, a);
    scene.draw(inst);
}

pub fn render() void {
    const dt = zupra.app.getDelta();
    const s = &scene.ssao.settings;

    if (zupra.input.isKeyJustPressed(._1)) scene.ssao_enabled = !scene.ssao_enabled;
    if (zupra.input.isKeyJustPressed(._9)) show_ao_buffer = !show_ao_buffer;
    if (zupra.input.isKeyJustPressed(._2)) s.radius = @max(0.05, s.radius - 0.1);
    if (zupra.input.isKeyJustPressed(._3)) s.radius = @min(8.0, s.radius + 0.1);
    if (zupra.input.isKeyJustPressed(._4)) s.intensity = @max(0.0, s.intensity - 0.1);
    if (zupra.input.isKeyJustPressed(._5)) s.intensity = @min(3.0, s.intensity + 0.1);
    if (zupra.input.isKeyJustPressed(._6) and s.slices > 1) s.slices -= 1;
    if (zupra.input.isKeyJustPressed(._7) and s.slices < 8) s.slices += 1;
    if (zupra.input.isKeyJustPressed(._8)) s.steps = if (s.steps >= 12) 2 else s.steps + 2;

    // Everything else off, so nothing downstream can be blamed for what AO does.
    if (zupra.input.isKeyJustPressed(._0)) {
        isolate = !isolate;
        scene.bloom_enabled = !isolate;
        scene.setAAMethod(if (isolate) .none else .taa);
    }

    controller.update(dt);
    controller.applyTo(&cam);

    scene.begin(cam, &env);

    place(&floor_model, 0, 0, 0);

    // --- REFERENCE REGION -------------------------------------------------
    // Everything below sits at z >= 0. The half of the floor at negative z is
    // deliberately EMPTY: nothing within any sane radius, so correct AO there is
    // exactly 1.0. Toggling AO must not change it at all. If it darkens, the bug
    // is in the open-surface path.
    // ----------------------------------------------------------------------

    // A single isolated sphere. The only thing that should darken is its contact
    // ring; its silhouette against empty floor is where a rim shows up with
    // nothing nearby to confuse the reading.
    place(&ball_model, -7.0, 1.2, 4.0);

    // An inside corner on its own, far from the rest.
    placeRot(&wall_model, 7.0, 3.0, 10.0, 0);
    placeRot(&wall_model, 13.0, 3.0, 4.0, std.math.pi * 0.5);

    // Three scales of 90-degree inside angle, to check the radius resolves
    // detail at the size it claims to. A radius of 0.6 should darken the small
    // block's seams and barely touch the large one's.
    place(&block_l, 0.0, 1.5, 8.0);
    place(&block_m, 2.5, 0.5, 8.0);
    place(&block_s, 4.0, 0.15, 8.0);
    // Each block also butted against a neighbour, forming a tight slot.
    place(&block_m, 2.5, 1.5, 8.0);
    place(&block_s, 4.0, 0.45, 8.0);

    scene.end();

    fps_accum += dt;
    fps_frames += 1;
    if (fps_accum >= 0.25) {
        const fps = @as(f32, @floatFromInt(fps_frames)) / fps_accum;
        fps_text = std.fmt.bufPrint(
            &fps_buf,
            "{d:.0} FPS {d:.2}ms | AO {s} r={d:.2} i={d:.2} slices={d} steps={d} | raw={s} isolate={s}  [1 ao 9 raw 2/3 r 4/5 i 6/7 slices 8 steps 0 isolate]",
            .{
                fps,                                     1000.0 / fps,
                if (scene.ssao_enabled) "ON" else "OFF", s.radius,
                s.intensity,                             s.slices,
                s.steps,                                 if (show_ao_buffer) "on" else "off",
                if (isolate) "on" else "off",
            },
        ) catch "FPS ?";
        fps_accum = 0;
        fps_frames = 0;
    }

    ui_cam.setViewport(sokol.app.widthf(), sokol.app.heightf());
    zupra.beginDrawing();

    if (show_ao_buffer) {
        var ao_sprite = Sprite.init(TextureRegion.full(scene.ssao.debugTexture()));
        ao_sprite.dest = .{ .x = 0, .y = 0, .width = sokol.app.widthf(), .height = sokol.app.heightf() };
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
    ball_model.deinit();
    block_l.deinit();
    block_m.deinit();
    block_s.deinit();
    env.deinit();
    scene.deinit();
    cache.deinit();
}
