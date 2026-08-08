//! src/render/ibl.zig
//!
//! Image-Based Lighting bake. Two stages:
//!   1. Render the environment (the skybox) into a cubemap (env_cube) — one
//!      fullscreen pass per face, reusing the sky shader.
//!   2. Convolve env_cube over the hemisphere into a small irradiance cubemap —
//!      the diffuse ambient term the lighting shaders will sample.
//!
//!   3. Prefilter env_cube into a MIPPED specular cubemap (prefilter_cube): each
//!      mip is the environment GGX-importance-sampled for one roughness level
//!      (mip 0 = mirror, higher mips = rougher). This is the specular IBL.
//!
//! Baking stage 1 from a cubemap (rather than sampling the sky procedurally in
//! the convolution) keeps the convolution source-agnostic: a loaded cubemap or
//! HDR-derived env_cube convolves identically later.

const std = @import("std");
const sg = @import("sokol").gfx;
const zupra = @import("../root.zig");
const math = @import("../math.zig");
const zm = math.zm;
const pipeline = @import("../graphics/pipeline.zig");
const cubemap = @import("cubemap.zig");
const Cubemap = cubemap.Cubemap;
const FullscreenTriangle = @import("fullscreen.zig").FullscreenTriangle;
const Skybox = @import("skybox.zig").Skybox;
const EnvironmentMap = @import("render.zig").EnvironmentMap;

const shd_irr = @import("shaders").irradiance;
const shd_pre = @import("shaders").prefilter;

const PipelineCache = pipeline.PipelineCache;
const PipelineKey = pipeline.PipelineKey;
const PassSignature = pipeline.PassSignature;
const Matrix = math.Matrix;
const Vec3 = math.Vec3;

const env_size = 128; // env cubemap face size (source for convolution + prefilter mip0)
const irradiance_size = 32; // irradiance is low-frequency: small is fine
const prefilter_size = 128; // specular prefilter base face size
const prefilter_mips = 5; // roughness levels: 0, 0.25, 0.5, 0.75, 1.0

const ConvParams = extern struct {
    inv_view_proj: [16]f32,
};

const PreParams = extern struct {
    inv_view_proj: [16]f32,
    params: [4]f32, // x = roughness
};

pub const Ibl = struct {
    cache: *PipelineCache,
    env_cube: Cubemap,
    irradiance_cube: Cubemap,
    prefilter_cube: Cubemap,
    sampler: sg.Sampler, // linear + mipmap, clamp on all axes
    irr_shader: sg.Shader,
    pre_shader: sg.Shader,
    brdf_lut_img: sg.Image,
    brdf_lut_view: sg.View,
    tri: FullscreenTriangle,

    pub fn init(allocator: std.mem.Allocator, cache: *PipelineCache) Ibl {
        const brdf = loadBrdfLut(allocator);
        return .{
            .cache = cache,
            .brdf_lut_img = brdf.img,
            .brdf_lut_view = brdf.view,
            .env_cube = Cubemap.initRenderTarget(env_size, .RGBA16F, 1),
            .irradiance_cube = Cubemap.initRenderTarget(irradiance_size, .RGBA16F, 1),
            .prefilter_cube = Cubemap.initRenderTarget(prefilter_size, .RGBA16F, prefilter_mips),
            .sampler = sg.makeSampler(.{
                .min_filter = .LINEAR,
                .mag_filter = .LINEAR,
                .mipmap_filter = .LINEAR, // interpolate between roughness mips
                .wrap_u = .CLAMP_TO_EDGE,
                .wrap_v = .CLAMP_TO_EDGE,
                .wrap_w = .CLAMP_TO_EDGE,
            }),
            .irr_shader = sg.makeShader(shd_irr.irradianceShaderDesc(sg.queryBackend())),
            .pre_shader = sg.makeShader(shd_pre.prefilterShaderDesc(sg.queryBackend())),
            .tri = FullscreenTriangle.init(),
        };
    }

    pub fn deinit(self: *Ibl) void {
        self.tri.deinit();
        sg.destroyView(self.brdf_lut_view);
        sg.destroyImage(self.brdf_lut_img);
        sg.destroyShader(self.pre_shader);
        sg.destroyShader(self.irr_shader);
        sg.destroySampler(self.sampler);
        self.prefilter_cube.deinit();
        self.irradiance_cube.deinit();
        self.env_cube.deinit();
    }

    /// Run both bake stages. Call inside a frame (passes need a commit), once —
    /// the result is static unless the environment changes.
    pub fn bake(self: *Ibl, skybox: *Skybox, envmap: ?*EnvironmentMap) void {
        const face_vps = cubemap.faceViewProjections();
        const origin = Vec3{ .x = 0, .y = 0, .z = 0 };

        // Stage 1: sky -> env_cube (no depth, fill each face).
        const env_sig = faceSig(self.env_cube.format);
        for (0..6) |face| {
            const inv_vp = zm.inverse(face_vps[face]);
            zupra.beginDrawingPass(facePass(self.env_cube.faceAttachment(@intCast(face), 0)));
            if (envmap) |em| {
                em.renderRaw(inv_vp, origin, env_sig, false);
            } else {
                skybox.renderRaw(inv_vp, origin, env_sig, false);
            }
            zupra.endDrawing();
        }

        // Stage 2: env_cube -> irradiance_cube (convolution).
        const irr_sig = faceSig(self.irradiance_cube.format);
        for (0..6) |face| {
            const inv_vp = zm.inverse(face_vps[face]);
            zupra.beginDrawingPass(facePass(self.irradiance_cube.faceAttachment(@intCast(face), 0)));
            self.convolveFace(inv_vp, irr_sig);
            zupra.endDrawing();
        }

        // Stage 3: env_cube -> prefilter_cube, one mip per roughness level.
        const pre_sig = faceSig(self.prefilter_cube.format);
        for (0..prefilter_mips) |mip| {
            const roughness = @as(f32, @floatFromInt(mip)) / @as(f32, prefilter_mips - 1);
            for (0..6) |face| {
                const inv_vp = zm.inverse(face_vps[face]);
                zupra.beginDrawingPass(facePass(self.prefilter_cube.faceAttachment(@intCast(face), @intCast(mip))));
                self.prefilterFace(inv_vp, roughness, pre_sig);
                zupra.endDrawing();
            }
        }
    }

    fn prefilterFace(self: *Ibl, inv_vp: Matrix, roughness: f32, sig: PassSignature) void {
        const key = PipelineKey{
            .shader = self.pre_shader,
            .layout = .fullscreen,
            .index_type = .u32,
            .indexed = false,
            .pass = sig,
            .primitive = .TRIANGLES,
            .cull = .NONE,
            .blend = .none,
            .depth_test = false,
            .depth_write = false,
        };
        const pip = self.cache.get(key) catch |err| {
            std.log.err("Ibl: prefilter pipeline failed: {}", .{err});
            return;
        };

        var params = PreParams{ .inv_view_proj = @bitCast(inv_vp), .params = .{ roughness, 0, 0, 0 } };

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = self.tri.vbuf;
        bindings.views[shd_pre.VIEW_env_cube] = self.env_cube.sample_view;
        bindings.samplers[shd_pre.SMP_smp] = self.sampler;

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(shd_pre.UB_pre_params, sg.asRange(&params));
        sg.draw(0, 3, 1);
    }

    fn convolveFace(self: *Ibl, inv_vp: Matrix, sig: PassSignature) void {
        const key = PipelineKey{
            .shader = self.irr_shader,
            .layout = .fullscreen,
            .index_type = .u32,
            .indexed = false,
            .pass = sig,
            .primitive = .TRIANGLES,
            .cull = .NONE,
            .blend = .none,
            .depth_test = false,
            .depth_write = false,
        };
        const pip = self.cache.get(key) catch |err| {
            std.log.err("Ibl: convolve pipeline failed: {}", .{err});
            return;
        };

        var params = ConvParams{ .inv_view_proj = @bitCast(inv_vp) };

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = self.tri.vbuf;
        bindings.views[shd_irr.VIEW_env_cube] = self.env_cube.sample_view;
        bindings.samplers[shd_irr.SMP_smp] = self.sampler;

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(shd_irr.UB_conv_params, sg.asRange(&params));
        sg.draw(0, 3, 1);
    }

    /// The irradiance cubemap view + sampler, for the lighting shaders.
    pub fn irradianceView(self: Ibl) sg.View {
        return self.irradiance_cube.sample_view;
    }
    pub fn cubeSampler(self: Ibl) sg.Sampler {
        return self.sampler;
    }

    /// The prefiltered specular cubemap view (mipped), for the lighting shaders.
    pub fn prefilterView(self: Ibl) sg.View {
        return self.prefilter_cube.sample_view;
    }
    /// Highest mip index (= roughness 1.0). Lighting scales roughness by this.
    pub fn prefilterMaxMip(self: Ibl) f32 {
        _ = self;
        return @as(f32, prefilter_mips - 1);
    }

    /// The split-sum BRDF lookup texture (RG16F), for the lighting shaders.
    pub fn brdfLutView(self: Ibl) sg.View {
        return self.brdf_lut_view;
    }
};

const brdf_lut_size = 512; // must match tools/brdflut_backer.zig LUT_SIZE
const brdf_lut_path = "resources/brdf.lut";

const BrdfLut = struct { img: sg.Image, view: sg.View };

/// Load the precomputed split-sum BRDF LUT (raw RG f16, row-major, no header).
/// On failure, fall back to a 1x1 (1,0) texel so the binding stays valid (the
/// specular IBL degrades but nothing crashes).
fn loadBrdfLut(allocator: std.mem.Allocator) BrdfLut {
    const bytes = readBrdfFile(allocator) catch |err| {
        std.log.err("Ibl: could not load {s}: {} — specular IBL will be approximate", .{ brdf_lut_path, err });
        return makeBrdfFallback();
    };
    defer allocator.free(bytes);

    const expected = brdf_lut_size * brdf_lut_size * 2 * 2; // RG * f16
    if (bytes.len != expected) {
        std.log.err("Ibl: {s} is {d} bytes, expected {d} — using fallback", .{ brdf_lut_path, bytes.len, expected });
        return makeBrdfFallback();
    }

    var desc = sg.ImageDesc{
        .width = brdf_lut_size,
        .height = brdf_lut_size,
        .pixel_format = .RG16F,
        .usage = .{ .immutable = true },
    };
    desc.data.mip_levels[0] = .{ .ptr = bytes.ptr, .size = bytes.len };
    const img = sg.makeImage(desc);
    return .{ .img = img, .view = sg.makeView(.{ .texture = .{ .image = img } }) };
}

fn readBrdfFile(allocator: std.mem.Allocator) ![]u8 {
    const io = zupra.getIo();
    const file = try std.Io.Dir.cwd().openFile(io, brdf_lut_path, .{});
    defer file.close(io);
    var file_reader = file.reader(io, &.{});
    return try file_reader.interface.allocRemaining(
        allocator,
        .limited(8 * 1024 * 1024),
    );
}

fn makeBrdfFallback() BrdfLut {
    // 1x1 RG16F = (1.0, 0.0): specular_ibl ~= prefiltered * F (scale only).
    const one: u16 = 0x3C00; // f16 1.0
    const zero: u16 = 0x0000; // f16 0.0
    const px = [2]u16{ one, zero };
    var desc = sg.ImageDesc{
        .width = 1,
        .height = 1,
        .pixel_format = .RG16F,
        .usage = .{ .immutable = true },
    };
    desc.data.mip_levels[0] = .{ .ptr = &px, .size = @sizeOf(@TypeOf(px)) };
    const img = sg.makeImage(desc);
    return .{ .img = img, .view = sg.makeView(.{ .texture = .{ .image = img } }) };
}

/// Single-color, no-depth pass signature for a cube face of the given format.
fn faceSig(format: sg.PixelFormat) PassSignature {
    var sig = PassSignature{ .color_count = 1, .depth_format = .NONE, .sample_count = 1 };
    sig.color_formats[0] = format;
    return sig;
}

fn facePass(attachment: sg.View) sg.Pass {
    var att = sg.Attachments{};
    att.colors[0] = attachment;
    var action = sg.PassAction{};
    action.colors[0] = .{ .load_action = .CLEAR, .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 1 } };
    return .{ .action = action, .attachments = att };
}
