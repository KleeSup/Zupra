//! examples/playground.zig
//!
//! Post-processing proof: render the 3D scene into a Framebuffer, then draw the
//! framebuffer fullscreen through a custom CRT post shader (shaders/post.glsl).
//! Exercises: FBO render -> sample as texture -> custom shader -> fs uniforms.

const std = @import("std");
const zupra = @import("zupra");
const sokol = zupra.intern.sokol;
const post_shd = @import("assets/shaders/post.glsl.zig");

const Sprite = zupra.graphics.texture.Sprite;
const PassSignature = zupra.graphics.pipeline.PassSignature;

var gpa: std.mem.Allocator = undefined;
var cache: zupra.graphics.pipeline.PipelineCache = undefined;
var mbatch: zupra.render.ModelBatch = undefined;
var sbatch: zupra.render.SpriteBatch = undefined;
var cam3d: zupra.render.Camera3D = undefined;
var cam2d: zupra.render.Camera2D = undefined;
var model: zupra.render.Model = undefined;
var fb: zupra.render.Framebuffer = undefined;
var post_shader: zupra.graphics.ShaderProgram = undefined;

var last_ticks: u64 = 0;
var t: f32 = 0;

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
    mbatch = zupra.render.ModelBatch.init(&cache);
    sbatch = zupra.render.SpriteBatch.init(gpa, &cache, .{}) catch unreachable;

    cam3d = zupra.render.Camera3D.init(16.0 / 9.0);
    cam3d.position = .{ .x = 3, .y = 2.5, .z = -5 };
    cam3d.target = .{ .x = 0, .y = 0, .z = 0 };

    cam2d = zupra.render.Camera2D.init(1280, 720);

    model = zupra.render.Model.fromMesh(gpa, zupra.render.MeshBuilder.cube(1, 1, 1), .{
        .base_color = .{ .r = 0.85, .g = 0.45, .b = 0.2, .a = 1 },
    }) catch unreachable;

    fb = zupra.render.Framebuffer.init(.{ .width = 1280, .height = 720, .sample_count = 1 });

    // The post shader has its own fragment uniform block (time), so set fs_params.
    post_shader = zupra.graphics.ShaderProgram.init(post_shd.postShaderDesc, .{
        .layout = .sprite,
        .slots = .{
            .tex_view = post_shd.VIEW_tex,
            .sampler = post_shd.SMP_smp,
            .vs_params = post_shd.UB_vs_params,
            .fs_params = post_shd.UB_fs_params,
        },
    });

    last_ticks = sokol.time.now();
}

pub fn render() void {
    const now = sokol.time.now();
    t += @floatCast(sokol.time.sec(sokol.time.diff(now, last_ticks)));
    last_ticks = now;

    // --- Pass 1: 3D scene into the framebuffer ---
    cam3d.setViewport(@floatFromInt(fb.width), @floatFromInt(fb.height));

    var inst = model.instance();
    inst.setRotationAxisAngle(.{ .x = 0, .y = 1, .z = 0 }, t);

    zupra.beginDrawingFramebufferClear(fb, .{ .r = 0.1, .g = 0.06, .b = 0.08, .a = 1 });
    mbatch.beginEx(cam3d, .{}, fb.passSignature());
    mbatch.draw(inst);
    mbatch.end();
    zupra.endDrawing();

    // --- Pass 2: framebuffer -> screen through the post shader ---
    const ww = sokol.app.widthf();
    const wh = sokol.app.heightf();
    cam2d.setViewport(ww, wh);

    var screen = Sprite.init(.full(fb.asTexture()));
    screen.dest = .{ .x = 0, .y = 0, .width = ww, .height = wh };
    screen.flip_y = !sokol.gfx.queryFeatures().origin_top_left;

    var fsp = post_shd.FsParams{ .params = .{ t, 0, 0, 0 } };

    zupra.beginDrawingClear(zupra.colors.BLACK);
    sbatch.beginEx(cam2d, .none, PassSignature.swapchainPass(), post_shader, std.mem.asBytes(&fsp));
    sbatch.draw(screen);
    sbatch.end();
    zupra.endDrawing();
}

pub fn deinit() void {
    post_shader.deinit();
    fb.deinit();
    model.deinit();
    sbatch.deinit(gpa);
    cache.deinit();
}
