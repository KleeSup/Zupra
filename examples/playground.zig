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
const Camera2D = zupra.render.Camera2D;

var gpa: std.mem.Allocator = undefined;
var cache: zupra.graphics.pipeline.PipelineCache = undefined;
var scene: zupra.render.SceneRenderer = undefined;
var cam: zupra.render.Camera3D = undefined;
var env: Environment = .{};
var controller: zupra.render.FirstPersonController = undefined;
var font: zupra.render.Font = undefined;
var batch: zupra.render.SpriteBatch = undefined;
var ui_cam: Camera2D = undefined;
var text: zupra.render.TextBatch2D = undefined;

var fps_accum: f32 = 0;
var fps_frames: u32 = 0;
var fps_buf: [64]u8 = undefined;
var fps_text: []const u8 = "";

var floor_model: Model = undefined;
var metal_model: Model = undefined;
var rough_model: Model = undefined;
var glass_model: Model = undefined;
var wall_model: Model = undefined;
var helmet_model: Model = undefined;

var model_queue: std.ArrayList(Model) = .empty;

var last_ticks: u64 = 0;
var t: f32 = 0;

const holo_shader = @import("shaders").test_hologram;
var holo_prog: zupra.graphics.ShaderProgram = undefined;

const chroma_shader = @import("assets/shaders/grade.glsl.zig");
var program: zupra.graphics.ShaderProgram = undefined;
var effect: *zupra.render.PostEffect = undefined;

pub fn main(ctx: std.process.Init) !void {
    gpa = ctx.gpa;
    zupra.init(ctx, .{ .initFn = init, .renderFn = render, .deinitFn = deinit, .eventFn = onEvent });
}

pub fn init() void {
    cache = .init(gpa);
    scene = zupra.render.SceneRenderer.init(gpa, &cache, .deferred, 1280, 720);
    scene.setAAMethod(.none);

    cam = zupra.render.Camera3D.init(16.0 / 9.0);
    controller = .init(.{ .x = 0, .y = 0, .z = 0 });

    font = zupra.render.Font.initFromMemory(gpa, @embedFile("assets/OpenSans-Regular.ttf"), .{}) catch unreachable;
    batch = zupra.render.SpriteBatch.init(gpa, &cache, .{}) catch unreachable;
    ui_cam = Camera2D.init(1280, 720);
    text = zupra.render.TextBatch2D.init(&font, &batch);

    program = zupra.graphics.ShaderProgram.init(chroma_shader.gradeShaderDesc, .{
        .layout = .fullscreen,
        .slots = .{ .fs_params = chroma_shader.UB_fs_params },
    });

    effect = scene.post.addEffect(.{ .shader = program, .point = .ldr, .name = "chromatic" }) catch unreachable;
    effect.setUniforms(chroma_shader.FsParams{
        .tone = .{ 1.0, 1.0, 0.0, 0.0 }, // exposure, contrast, saturation=0 → greyscale
        .tint = .{ 1, 1, 1, 0.6 },
    });

    holo_prog = zupra.graphics.ShaderProgram.init(holo_shader.hologramShaderDesc, .{
        .layout = .mesh,
        .slots = .{ .fs_params = holo_shader.UB_fs_params },
        // no uv_params: this shader doesn't declare one
    });

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

    helmet_model = zupra.render.gltf.loadMemory(gpa, @embedFile("assets/models/CarConcept.glb")) catch unreachable;
    for (helmet_model.materials) |*m| {
        m.shader = holo_prog;
        m.alpha_mode = .blend; // it outputs alpha, so it wants blending
        m.setShaderUniforms(holo_shader.FsParams{
            .tint = .{ 0.2, 0.9, 1.0, 0.85 },
            .params = .{ 3.0, 18.0, 2.0, 0.0 },
        });
    }

    // Lighting: a warm sun + a colored point light.
    env = .{ .ambient = .{ .r = 0.03, .g = 0.03, .b = 0.04, .a = 1 } };
    env.addLight(Light.directional(.{ .x = -0.5, .y = -1.0, .z = -0.35 }, .{ .r = 1.0, .g = 0.96, .b = 0.9, .a = 1 }, 2.5));

    last_ticks = sokol.time.now();
}

pub fn render() void {
    if (zupra.input.isKeyJustPressed(._1)) { // raw
        scene.setAAMethod(.none);
        scene.setRenderScale(1);
        std.debug.print("Switched to: RAW\n", .{});
    } else if (zupra.input.isKeyJustPressed(._2)) { // FXAA
        scene.setAAMethod(.fxaa);
        scene.setRenderScale(1);
        std.debug.print("Switched to: FXAA\n", .{});
    } else if (zupra.input.isKeyJustPressed(._3)) { // better FXAA
        scene.setAAMethod(.fxaa_quality);
        scene.setRenderScale(1);
        std.debug.print("Switched to: FXAA_Q\n", .{});
    } else if (zupra.input.isKeyJustPressed(._4)) { // 2x SSAA
        scene.setAAMethod(.none);
        scene.setRenderScale(2);
        std.debug.print("Switched to: 2x SSAA\n", .{});
    } else if (zupra.input.isKeyJustPressed(._5)) { // both
        scene.setAAMethod(.fxaa);
        scene.setRenderScale(2);
        std.debug.print("Switched to: BOTH\n", .{});
    } else if (zupra.input.isKeyJustPressed(._0)) {
        if (scene.mode == .forward) {
            if (scene.msaa_samples == 1) {
                scene.setMsaa(4);
                std.debug.print("Switched MSAA: ON\n", .{});
            } else {
                scene.setMsaa(1);
                std.debug.print("Switched MSAA: OFF\n", .{});
            }
        }
    }

    const now = sokol.time.now();
    t += @floatCast(sokol.time.sec(sokol.time.diff(now, last_ticks)));
    last_ticks = now;

    for (helmet_model.materials) |*m| {
        m.setShaderUniforms(holo_shader.FsParams{
            .tint = .{ 0.2, 0.9, 1.0, 0.85 },
            .params = .{ 3.0, 18.0, 2.0, t },
        });
    }

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

    // Debug
    const dt = zupra.app.getDelta();
    fps_accum += dt;
    fps_frames += 1;
    if (fps_accum >= 0.25) { // refresh 4x/sec
        const fps = @as(f32, @floatFromInt(fps_frames)) / fps_accum;
        fps_text = std.fmt.bufPrint(&fps_buf, "{d:.0} FPS   {d:.2} ms", .{ fps, 1000.0 / fps }) catch "FPS ?";
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
    font.deinit();
    batch.deinit(gpa);
    floor_model.deinit();
    metal_model.deinit();
    rough_model.deinit();
    glass_model.deinit();
    wall_model.deinit();
    helmet_model.deinit();
    program.deinit();
    scene.deinit();
    cache.deinit();
}
