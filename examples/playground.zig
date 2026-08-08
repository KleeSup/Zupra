//! examples/playground.zig
//!
//! CLUSTERED LIGHTING + SHADOWS demo. A mixed grid of PBR spheres and slowly
//! spinning cubes under all three shadow-casting light types at once:
//!
//!   * a cascaded directional sun,
//!   * a sweeping spot light with its own perspective shadow map,
//!   * a point light in the middle of the grid casting in all six directions,
//!
//! plus as many unshadowed moving point lights as you care to add, to keep the
//! froxel grid honest.
//!
//! Watching all three together is the point: each light's shadow only removes
//! that light's contribution, so where the sun's shadow and the spot's shadow
//! cross you get a third, darker region rather than either cancelling out.
//!
//! Keys:
//!   6   add 16 point lights      7   remove 16
//!   8   toggle light motion      9   toggle the shadowed point light
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

/// One surface look. Six of them, each built once as a sphere and again as a
/// cube, so the grid can mix shape and material independently.
const Look = struct {
    color: zupra.Color,
    metallic: f32,
    roughness: f32,
};

const palette = [_]Look{
    .{ .color = .{ .r = 0.90, .g = 0.90, .b = 0.92, .a = 1 }, .metallic = 1.0, .roughness = 0.18 }, // polished steel
    .{ .color = .{ .r = 0.85, .g = 0.30, .b = 0.32, .a = 1 }, .metallic = 0.0, .roughness = 0.55 }, // red plastic
    .{ .color = .{ .r = 0.20, .g = 0.45, .b = 0.80, .a = 1 }, .metallic = 0.0, .roughness = 0.35 }, // blue enamel
    .{ .color = .{ .r = 0.95, .g = 0.75, .b = 0.25, .a = 1 }, .metallic = 1.0, .roughness = 0.32 }, // gold
    .{ .color = .{ .r = 0.25, .g = 0.65, .b = 0.40, .a = 1 }, .metallic = 0.0, .roughness = 0.75 }, // matte green
    .{ .color = .{ .r = 0.55, .g = 0.35, .b = 0.75, .a = 1 }, .metallic = 0.3, .roughness = 0.28 }, // violet
};

var sphere_models: [palette.len]Model = undefined;
var cube_models: [palette.len]Model = undefined;

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

var spot_light: LightHandle = undefined;
var point_light: LightHandle = undefined;
var point_shadow_on = true;

var rng = std.Random.DefaultPrng.init(0x9E3779B97F4A7C15);

var fps_accum: f32 = 0;
var fps_frames: u32 = 0;
var fps_buf: [192]u8 = undefined;
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
    scene = zupra.render.SceneRenderer.init(gpa, &cache, .forward, 1280, 720);
    scene.setAAMethod(.fxaa);

    cam = zupra.render.Camera3D.init(16.0 / 9.0);
    controller = .init(.{ .x = 0, .y = 6, .z = -18 });

    font = zupra.render.Font.initFromMemory(gpa, @embedFile("assets/OpenSans-Regular.ttf"), .{}) catch unreachable;
    batch = zupra.render.SpriteBatch.init(gpa, &cache, .{}) catch unreachable;
    ui_cam = Camera2D.init(1280, 720);
    text = zupra.render.TextBatch2D.init(&font, &batch);

    env = Environment.init(gpa);
    env.ambient = .{ .r = 0.03, .g = 0.03, .b = 0.04, .a = 1 };

    // ---------------------------------------------------------------------
    //  Lights
    //
    //  ATLAS BUDGET. Every shadowed light competes for tiles in one 4096
    //  atlas, and a point light is six of them. This set comes to 4x1024
    //  (sun) + 1x1024 (spot) + 6x512 (point) = 11 tiles. Raising the sun to
    //  2048 would alone consume most of the atlas and starve the other two --
    //  which is exactly the trade a real scene has to make, so it's worth
    //  seeing here rather than hiding behind a bigger atlas.
    // ---------------------------------------------------------------------

    const sun = env.addLight(Light.directional(
        .{ .x = -0.5, .y = -1.0, .z = -0.35 },
        .{ .r = 1.0, .g = 0.96, .b = 0.9, .a = 1 },
        3.0,
    )) catch unreachable;
    if (env.getLight(sun)) |l| {
        l.shadow = .{ .enabled = true, .resolution = 1024, .cascade_count = 4, .max_distance = 60 };
    }

    // Sweeps a circle overhead each frame (see render). Aimed inward and down,
    // so its cone crosses the sun's shadows at a steep angle and the two sets
    // of shadows stay easy to tell apart.
    spot_light = env.addLight(Light.spot(
        .{ .x = -10, .y = 10, .z = -8 },
        .{ .x = 0.7, .y = -1.0, .z = 0.55 },
        .{ .r = 1.0, .g = 0.93, .b = 0.75, .a = 1 },
        150.0, // inverse-square falloff, so far higher than the sun's 3.0
        36.0, // range: also the shadow far plane and the clustering bound
        16.0,
        24.0, // outer angle IS the shadow frustum's half-angle
    )) catch unreachable;
    if (env.getLight(spot_light)) |l| {
        l.shadow = .{ .enabled = true, .resolution = 1024, .cascade_count = 1, .max_distance = 90 };
    }

    // Sits low in the middle of the grid so its six faces all have something to
    // cast: shadows radiate outward in every direction from one source.
    point_light = env.addLight(Light.point(
        .{ .x = 0, .y = 2.2, .z = 0 },
        .{ .r = 0.55, .g = 0.85, .b = 1.0, .a = 1 },
        60.0,
        14.0,
    )) catch unreachable;
    if (env.getLight(point_light)) |l| {
        // 512 per face, not 1024: six tiles at 1024 is a quarter of the atlas
        // for one light. Faces are only ever seen from inside the light's
        // range, so they need less resolution than they look like they should.
        l.shadow = .{ .enabled = true, .resolution = 512, .cascade_count = 1, .max_distance = 60 };
    }

    // ---------------------------------------------------------------------
    //  Geometry
    // ---------------------------------------------------------------------

    floor_model = Model.fromMesh(gpa, MeshBuilder.plane(48, 48), .{
        .base_color = .{ .r = 0.5, .g = 0.5, .b = 0.55, .a = 1 },
        .metallic = 0.0,
        .roughness = 0.9,
    }) catch unreachable;

    for (palette, 0..) |look, i| {
        const mat = Material{
            .base_color = look.color,
            .metallic = look.metallic,
            .roughness = look.roughness,
        };
        sphere_models[i] = Model.fromMesh(
            gpa,
            MeshBuilder.sphere(gpa, 0.9, 32, 32) catch unreachable,
            mat,
        ) catch unreachable;
        cube_models[i] = Model.fromMesh(gpa, MeshBuilder.cube(1.5, 1.5, 1.5), mat) catch unreachable;
    }
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

    // Toggling this is the clearest way to see what six tiles actually buy:
    // shadows radiating from the centre appear and vanish, and the caster count
    // in the overlay jumps by six.
    if (zupra.input.isKeyJustPressed(._9)) {
        point_shadow_on = !point_shadow_on;
        if (env.getLight(point_light)) |l| l.shadow.enabled = point_shadow_on;
    }

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

    if (lights_move) {
        // Orbit + bob each unshadowed point light, via its stable handle.
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

        // Walk the spot around the grid, always aimed at the centre. Moving the
        // light rather than the geometry is the honest test of the shadow map:
        // the whole frustum is refitted every frame, so any instability in the
        // projection shows up as swimming shadows.
        if (env.getLight(spot_light)) |l| {
            const a = elapsed * 0.35;
            const px = @cos(a) * 13.0;
            const pz = @sin(a) * 13.0;
            l.position = .{ .x = px, .y = 11.0, .z = pz };
            l.direction = .{ .x = -px, .y = -11.0, .z = -pz };
        }
    }

    controller.update(dt);
    controller.applyTo(&cam);

    scene.begin(cam, &env);

    var floor = floor_model.instance();
    floor.setPosition(.{ .x = 0, .y = 0, .z = 0 });
    scene.draw(floor);

    var gz: usize = 0;
    while (gz < grid_z) : (gz += 1) {
        var gx: usize = 0;
        while (gx < grid_x) : (gx += 1) {
            const px = (@as(f32, @floatFromInt(gx)) - grid_x * 0.5 + 0.5) * grid_spacing;
            const pz = (@as(f32, @floatFromInt(gz)) - grid_z * 0.5 + 0.5) * grid_spacing;

            // Coprime strides so shape and colour cycle at different rates and
            // the grid doesn't fall into an obvious stripe.
            const look = (gx * 5 + gz * 3) % palette.len;
            const is_cube = (gx + gz * 3) % 3 == 0;

            if (is_cube) {
                var inst = cube_models[look].instance();
                inst.setPosition(.{ .x = px, .y = 0.75, .z = pz });
                // A turning box is the best shadow test in the scene: its
                // silhouette changes every frame, so a stale or cached map
                // would be obvious immediately.
                inst.setRotationAxisAngle(
                    .{ .x = 0, .y = 1, .z = 0 },
                    elapsed * 0.5 + @as(f32, @floatFromInt(gx + gz)),
                );
                scene.draw(inst);
            } else {
                var inst = sphere_models[look].instance();
                inst.setPosition(.{ .x = px, .y = 0.9, .z = pz });
                scene.draw(inst);
            }
        }
    }
    scene.end();

    fps_accum += dt;
    fps_frames += 1;
    if (fps_accum >= 0.25) {
        const fps = @as(f32, @floatFromInt(fps_frames)) / fps_accum;
        // Caster count is the shadow system's real cost signal: one depth pass
        // per caster per frame. Sun cascades + spot + point faces, so it should
        // read 11 with everything on and 5 with the point light's shadow off.
        // shadow_draws is the depth draws that survived per-view culling; the
        // uncalled number is queue length times caster count, so the two
        // together show exactly what the frustum test is rejecting.
        // Three numbers, and the gaps between them are the story: draws is what
        // the GPU is actually told to do, instances is what survived culling,
        // and uncalled is what a naive loop would have submitted.
        const uncalled = scene.shadow_queue.items.len * scene.shadows.casters.items.len;
        fps_text = std.fmt.bufPrint(&fps_buf, "{d:.0} FPS {d:.2} ms  lights: {d}  visible: {d}/{d}  casters: {d}  shadow: {d} ({d}/{d})", .{
            fps,                 1000.0 / fps,           env.lightCount(),
            scene.visible_draws, scene.submitted_draws,  scene.shadows.casters.items.len,
            scene.shadow_draws,  scene.shadow_instances, uncalled,
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
    for (&sphere_models) |*m| m.deinit();
    for (&cube_models) |*m| m.deinit();
    env.deinit();
    scene.deinit();
    cache.deinit();
}
