//! examples/playground.zig
//!
//! CLUSTERED LIGHTING demo. A grid of PBR spheres under a dim sun plus MANY
//! moving point lights - the thing a 16-light uniform array could never do.
//! Lights orbit and bob so you can watch clustering track them live.
//!
//! Keys:
//!   6   add 16 point lights      7   remove 16
//!   8   toggle light motion
//!   1..5  AA (raw / FXAA / FXAA-quality / SSAA / both)   0  AA off

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

var floor_model: Model = undefined;
var sphere_metal: Model = undefined;
var sphere_rough: Model = undefined;

const grid_x = 8;
const grid_z = 8;
const grid_spacing: f32 = 3.0;

const PointMover = struct {
    handle: LightHandle,
    center: [3]f32,
    radius: f32,
    speed: f32,
    phase: f32,
    height: f32,
    bob: f32,
};
var movers: std.ArrayList(PointMover) = .empty;
var lights_move = true;
var elapsed: f32 = 0;

var rng = std.Random.DefaultPrng.init(0x9E3779B97F4A7C15);

var fps_accum: f32 = 0;
var fps_frames: u32 = 0;
var fps_buf: [96]u8 = undefined;
var fps_text: []const u8 = "";

pub fn main(ctx: std.process.Init) !void {
    zupra.init(ctx, .{ .initFn = init, .renderFn = render, .deinitFn = deinit, .eventFn = onEvent });
}

fn randF(lo: f32, hi: f32) f32 {
    return lo + rng.random().float(f32) * (hi - lo);
}

fn randColor() zupra.Color {
    return .{ .r = randF(0.2, 1.0), .g = randF(0.2, 1.0), .b = randF(0.2, 1.0), .a = 1 };
}

fn addMover() void {
    const center = [3]f32{
        randF(-grid_x * grid_spacing * 0.5, grid_x * grid_spacing * 0.5),
        randF(1.0, 3.0),
        randF(-grid_z * grid_spacing * 0.5, grid_z * grid_spacing * 0.5),
    };
    const handle = env.addLight(Light.point(
        .{ .x = center[0], .y = center[1], .z = center[2] },
        randColor(),
        randF(8.0, 16.0),
        randF(4.0, 7.0), // range = the clustering cull bound; keep it tight
    )) catch return;

    movers.append(gpa, .{
        .handle = handle,
        .center = center,
        .radius = randF(1.5, 5.0),
        .speed = randF(0.3, 1.2) * (if (rng.random().boolean()) @as(f32, 1) else -1),
        .phase = randF(0, std.math.tau),
        .height = center[1],
        .bob = randF(0.3, 1.2),
    }) catch {};
}

fn removeMovers(n: usize) void {
    var i: usize = 0;
    while (i < n and movers.items.len > 0) : (i += 1) {
        const m = movers.pop().?;
        env.removeLight(m.handle);
    }
}

pub fn init() void {
    gpa = zupra.getGPA();
    cache = .init(gpa);
    scene = zupra.render.SceneRenderer.init(gpa, &cache, .deferred, 1280, 720);
    scene.setAAMethod(.fxaa);

    cam = zupra.render.Camera3D.init(16.0 / 9.0);
    controller = .init(.{ .x = 0, .y = 6, .z = -18 });

    font = zupra.render.Font.initFromMemory(gpa, @embedFile("assets/OpenSans-Regular.ttf"), .{}) catch unreachable;
    batch = zupra.render.SpriteBatch.init(gpa, &cache, .{}) catch unreachable;
    ui_cam = Camera2D.init(1280, 720);
    text = zupra.render.TextBatch2D.init(&font, &batch);

    // Environment now owns GPU resources (clustered light store), so it takes an
    // allocator and must be passed by pointer, never copied.
    env = Environment.init(gpa);
    env.ambient = .{ .r = 0.02, .g = 0.02, .b = 0.03, .a = 1 };

    // Dim sun so the point lights dominate.
    const sun = env.addLight(Light.directional(
        .{ .x = -0.5, .y = -1.0, .z = -0.35 },
        .{ .r = 1.0, .g = 0.96, .b = 0.9, .a = 1 },
        3.0,
    )) catch unreachable;

    if (env.getLight(sun)) |l| {
        l.shadow = .{
            .enabled = true,
            //     .resolution = 2048,
            //.cascade_count = 4,
            .max_distance = 60,
        };
    }

    // Spot shadow test rig. Positioned high and off to one side so the cone
    // hits the sphere grid at a shallow angle -- the case that exposes a bad
    // near plane or bad bias fastest.
    const spot = env.addLight(Light.spot(
        .{ .x = -8, .y = 9, .z = -6 }, // position
        .{ .x = 0.6, .y = -1.0, .z = 0.45 }, // cone axis (travel direction)
        .{ .r = 1.0, .g = 0.95, .b = 0.8, .a = 1 },
        120.0, // intensity: punctual falloff is inverse-square, so this
        // needs to be far higher than a directional's to read
        30.0, // range: also the clustering cull bound, keep it honest
        18.0, // inner cone half-angle, full intensity within
        26.0, // outer half-angle, faded to zero -- and the shadow frustum
    )) catch unreachable;

    if (env.getLight(spot)) |l| {
        // cascade_count stays 1: a spot's cone already is the frustum, so there
        // is nothing to split. max_distance is the cull range for the light
        // itself here, not a cascade range.
        l.shadow = .{ .enabled = true, .resolution = 1024, .cascade_count = 1, .max_distance = 80 };
    }

    floor_model = Model.fromMesh(gpa, MeshBuilder.plane(48, 48), .{
        .base_color = .{ .r = 0.5, .g = 0.5, .b = 0.55, .a = 1 },
        .metallic = 0.0,
        .roughness = 0.9,
    }) catch unreachable;

    sphere_metal = Model.fromMesh(gpa, MeshBuilder.sphere(gpa, 0.9, 32, 32) catch unreachable, .{
        .base_color = .{ .r = 0.9, .g = 0.9, .b = 0.9, .a = 1 },
        .metallic = 1.0,
        .roughness = 0.25,
    }) catch unreachable;
    sphere_rough = Model.fromMesh(gpa, MeshBuilder.sphere(gpa, 0.9, 32, 32) catch unreachable, .{
        .base_color = .{ .r = 0.8, .g = 0.3, .b = 0.3, .a = 1 },
        .metallic = 0.0,
        .roughness = 0.6,
    }) catch unreachable;

    // 48 lights to start - already 3x the old fixed cap.
    var i: usize = 0;
    while (i < 63) : (i += 1) addMover();
}

pub fn render() void {
    const dt = zupra.app.getDelta();
    if (lights_move) elapsed += dt;

    if (zupra.input.isKeyJustPressed(._6)) {
        var i: usize = 0;
        while (i < 16) : (i += 1) addMover();
    }
    if (zupra.input.isKeyJustPressed(._7)) removeMovers(16);
    if (zupra.input.isKeyJustPressed(._8)) lights_move = !lights_move;

    if (zupra.input.isKeyJustPressed(._1)) {
        scene.setAAMethod(.none);
        scene.setRenderScale(1);
    } else if (zupra.input.isKeyJustPressed(._2)) {
        scene.setAAMethod(.fxaa);
        scene.setRenderScale(1);
    } else if (zupra.input.isKeyJustPressed(._3)) {
        scene.setAAMethod(.fxaa_quality);
        scene.setRenderScale(1);
    } else if (zupra.input.isKeyJustPressed(._4)) {
        scene.setAAMethod(.none);
        scene.setRenderScale(2);
    } else if (zupra.input.isKeyJustPressed(._5)) {
        scene.setAAMethod(.fxaa);
        scene.setRenderScale(2);
    } else if (zupra.input.isKeyJustPressed(._0)) {
        scene.setAAMethod(.none);
        scene.setRenderScale(1);
    }

    // Orbit + bob each point light. Mutated via the stable handle.
    if (lights_move) {
        for (movers.items) |m| {
            if (env.getLight(m.handle)) |l| {
                const a = m.phase + elapsed * m.speed;
                l.position = .{
                    .x = m.center[0] + @cos(a) * m.radius,
                    .y = m.height + @sin(elapsed * 1.7 + m.phase) * m.bob,
                    .z = m.center[2] + @sin(a) * m.radius,
                };
            }
        }
    }

    controller.update(dt);
    controller.applyTo(&cam);

    scene.begin(cam, &env); // pointer

    var floor = floor_model.instance();
    floor.setPosition(.{ .x = 0, .y = 0, .z = 0 });
    scene.draw(floor);

    var gz: usize = 0;
    while (gz < grid_z) : (gz += 1) {
        var gx: usize = 0;
        while (gx < grid_x) : (gx += 1) {
            const px = (@as(f32, @floatFromInt(gx)) - grid_x * 0.5 + 0.5) * grid_spacing;
            const pz = (@as(f32, @floatFromInt(gz)) - grid_z * 0.5 + 0.5) * grid_spacing;
            var inst = if ((gx + gz) & 1 == 0) sphere_metal.instance() else sphere_rough.instance();
            inst.setPosition(.{ .x = px, .y = 1.0, .z = pz });
            scene.draw(inst);
        }
    }
    scene.end();

    fps_accum += dt;
    fps_frames += 1;
    if (fps_accum >= 0.25) {
        const fps = @as(f32, @floatFromInt(fps_frames)) / fps_accum;
        fps_text = std.fmt.bufPrint(&fps_buf, "{d:.0} FPS  {d:.2} ms   lights: {d}   6/7=+/-16  8=pause", .{
            fps, 1000.0 / fps, env.lightCount(),
        }) catch "FPS ?";
        fps_accum = 0;
        fps_frames = 0;
    }

    ui_cam.setViewport(sokol.app.widthf(), sokol.app.heightf());
    zupra.beginDrawing();
    text.begin(ui_cam);
    text.draw(fps_text, 12, 12, 0.4, zupra.colors.WHITE);
    text.end();
    zupra.endDrawing();
}

pub fn onEvent(event: *const zupra.Event) void {
    controller.handleEvent(event);
}

pub fn deinit() void {
    movers.deinit(gpa);
    font.deinit();
    batch.deinit(gpa);
    floor_model.deinit();
    sphere_metal.deinit();
    sphere_rough.deinit();
    env.deinit();
    scene.deinit();
    cache.deinit();
}
