//! src/render/envmap.zig
//!
//! Equirectangular HDR environment map — a captured sky that replaces the
//! procedural one.
//!
//! This is the single largest quality step available to the PBR path, and the
//! reason is worth stating: the IBL bake chain (irradiance, prefiltered
//! specular, BRDF LUT) is already complete and correct, but it has only ever
//! been fed a smooth procedural gradient. A mirror sphere reflecting a gradient
//! looks like plastic no matter how right the BRDF is, because there is nothing
//! in the reflection to recognise. Feed the same chain a real capture and the
//! identical code produces convincing metal.
//!
//! One texture, two jobs. `renderRaw` reconstructs a world direction per pixel
//! from an inverse view-projection and samples the equirectangular map, which
//! serves both as the on-screen background (full resolution, depth-tested) and
//! as the source for the IBL bake's six cube faces (no depth, fill the face).
//! Ibl.bake was written source-agnostic -- it convolves whatever landed in
//! env_cube -- so nothing downstream of that first stage changes.
//!
//! FORMAT: zstbi detects Radiance .hdr and loads it as float, then converts to
//! f16 in place, so its buffer already matches sokol's RGBA16F byte for byte.
//! No conversion here, and half the footprint of RGBA32F -- which matters, since
//! these are commonly 4K or 8K wide.
//!
//! TWO SOURCES: an equirectangular map (usually HDR) or six cubemap faces
//! (usually PNG). Same renderer, different lookup, so they need different
//! programs -- a shader can't be polymorphic over texture type. Which one is in
//! use is decided at load and never branches per pixel.
//!
//! LDR cube faces are allowed but warned about. They are fine as scenery behind
//! a scene lit some other way, and poor as an IBL source: 8-bit clips at 1.0, so
//! the brightest part of the sky carries no more energy than white paper and the
//! specular prefilter has nothing to gather.

const std = @import("std");
const sg = @import("sokol").gfx;
const zstbi = @import("zstbi");
const zupra = @import("../root.zig");
const math = @import("../math.zig");
const pipeline = @import("../graphics/pipeline.zig");
const FullscreenTriangle = @import("fullscreen.zig").FullscreenTriangle;
const Camera3D = @import("camera3d.zig").Camera3D;

const shd = @import("shaders").sky_equirect;
const shd_cube = @import("shaders").sky_cube;

const PipelineCache = pipeline.PipelineCache;
const PipelineKey = pipeline.PipelineKey;
const PassSignature = pipeline.PassSignature;
const Matrix = math.Matrix;
const Vec3 = math.Vec3;

/// Identical layout in both programs, so one value serves either.
const EnvParams = extern struct {
    inv_view_proj: [16]f32,
    camera_pos: [4]f32,
    params: [4]f32, // x = intensity
};

pub const Source = enum { equirect, cube };

pub const EnvironmentMap = struct {
    cache: *PipelineCache,
    source: Source,
    /// Only the program matching `source` is created; the other would be a
    /// compiled shader nothing ever binds.
    shader: sg.Shader,
    tri: FullscreenTriangle,

    img: sg.Image,
    view: sg.View,
    sampler: sg.Sampler,
    width: u32,
    height: u32,

    /// Multiplies the sampled radiance. Captures are authored at arbitrary
    /// absolute scales, so this is the knob that makes one sit correctly against
    /// the scene's analytic lights rather than washing them out or vanishing.
    intensity: f32 = 1.0,

    /// Load from an encoded .hdr in memory (@embedFile, or bytes read by the
    /// caller). Four components forced: sokol has no RGB16F, and the alpha costs
    /// nothing that the row padding wouldn't.
    pub fn initFromMemory(cache: *PipelineCache, bytes: []const u8) !EnvironmentMap {
        var image = try zstbi.Image.loadFromMemory(bytes, 4);
        defer image.deinit();
        return initFromImage(cache, &image);
    }

    /// Load from a path on disk. Prefer initFromMemory with @embedFile for
    /// anything shipped -- this exists for tools and asset browsing, where the
    /// file is chosen at runtime.
    pub fn initFromFile(cache: *PipelineCache, path: [:0]const u8) !EnvironmentMap {
        var image = try zstbi.Image.loadFromFile(path, 4);
        defer image.deinit();
        return initFromImage(cache, &image);
    }

    /// Load from six encoded face images, in sokol's order:
    /// +X, -X, +Y, -Y, +Z, -Z (right, left, top, bottom, front, back).
    ///
    /// All six must share `size` and be square. Faces are decoded to RGBA8 and
    /// uploaded as **SRGB8A8**, not RGBA8 -- PNGs are authored in sRGB, and
    /// handing them to the renderer as linear makes the sky read washed out and
    /// far too bright, which then propagates into the IBL bake. Declaring the
    /// format lets the sampler linearise in hardware for free. (Texture
    /// .loadCubemapFromMemory hardcodes RGBA8, which is why this builds the
    /// image itself rather than calling it.)
    pub fn initFromCubeFaces(
        cache: *PipelineCache,
        alloc: std.mem.Allocator,
        faces: [6][]const u8,
        size: u32,
    ) !EnvironmentMap {
        const face_bytes = size * size * 4;
        const pixels = try alloc.alloc(u8, face_bytes * 6);
        defer alloc.free(pixels); // sokol copies immutable data during makeImage

        for (faces, 0..) |encoded, i| {
            var face = try zstbi.Image.loadFromMemory(encoded, 4);
            defer face.deinit();
            if (face.width != size or face.height != size) {
                std.log.err(
                    "EnvironmentMap: cube face {d} is {d}x{d}, expected {d}x{d}",
                    .{ i, face.width, face.height, size, size },
                );
                return error.FaceSizeMismatch;
            }
            @memcpy(pixels[face_bytes * i .. face_bytes * (i + 1)], face.data);
        }

        std.log.warn(
            "EnvironmentMap: cube faces are 8-bit, so highlights clip at 1.0. Fine as a background; expect a flat IBL bake. An .hdr equirectangular map is the better light source.",
            .{},
        );

        var desc = sg.ImageDesc{
            .type = .CUBE,
            .width = @intCast(size),
            .height = @intCast(size),
            .num_slices = 6,
            .pixel_format = .SRGB8A8,
            .usage = .{ .immutable = true },
        };
        desc.data.mip_levels[0] = .{ .ptr = pixels.ptr, .size = pixels.len };
        const img = sg.makeImage(desc);

        return .{
            .cache = cache,
            .source = .cube,
            .shader = sg.makeShader(shd_cube.skyCubeShaderDesc(sg.queryBackend())),
            .tri = FullscreenTriangle.init(),
            .img = img,
            .view = sg.makeView(.{ .texture = .{ .image = img } }),
            // Clamp on all three axes: sampling across a cube edge must not wrap
            // to the far side of the face.
            .sampler = sg.makeSampler(.{
                .min_filter = .LINEAR,
                .mag_filter = .LINEAR,
                .wrap_u = .CLAMP_TO_EDGE,
                .wrap_v = .CLAMP_TO_EDGE,
                .wrap_w = .CLAMP_TO_EDGE,
            }),
            .width = size,
            .height = size,
        };
    }

    fn initFromImage(cache: *PipelineCache, image: *zstbi.Image) !EnvironmentMap {
        // Refuse LDR input rather than silently producing a flat-looking bake.
        // An 8-bit source clips every value at 1.0, so the sun -- the entire
        // reason to use a capture -- carries no more energy than white paper,
        // and the specular prefilter has nothing bright to gather.
        if (!image.is_hdr) {
            std.log.err(
                "EnvironmentMap: source is not HDR (8-bit). Highlights are clipped at 1.0, so IBL will look flat. Use a Radiance .hdr.",
                .{},
            );
            return error.NotHdr;
        }

        var desc = sg.ImageDesc{
            .width = @intCast(image.width),
            .height = @intCast(image.height),
            .pixel_format = .RGBA16F,
            .usage = .{ .immutable = true },
        };
        desc.data.mip_levels[0] = .{ .ptr = image.data.ptr, .size = image.data.len };
        const img = sg.makeImage(desc);

        return .{
            .cache = cache,
            .source = .equirect,
            .shader = sg.makeShader(shd.skyEquirectShaderDesc(sg.queryBackend())),
            .tri = FullscreenTriangle.init(),
            .img = img,
            .view = sg.makeView(.{ .texture = .{ .image = img } }),
            // CLAMP vertically and REPEAT horizontally: longitude wraps around
            // the sphere, latitude does not. Clamping u instead would put a seam
            // behind the camera; repeating v would mirror the sky into the floor.
            .sampler = sg.makeSampler(.{
                .min_filter = .LINEAR,
                .mag_filter = .LINEAR,
                .wrap_u = .REPEAT,
                .wrap_v = .CLAMP_TO_EDGE,
            }),
            .width = image.width,
            .height = image.height,
        };
    }

    pub fn deinit(self: *EnvironmentMap) void {
        self.tri.deinit();
        sg.destroySampler(self.sampler);
        sg.destroyView(self.view);
        sg.destroyImage(self.img);
        sg.destroyShader(self.shader);
    }

    /// Draw as the screen background, depth-tested so geometry occludes it.
    pub fn render(self: *EnvironmentMap, camera: Camera3D, pass: PassSignature) void {
        self.renderRaw(math.zm.inverse(camera.viewProjection()), camera.position, pass, true);
    }

    /// Render for an explicit inverse view-projection. `depth_test` = true for
    /// the screen pass, false when filling an IBL cube face (no depth
    /// attachment, and the whole face must be covered).
    ///
    /// Signature deliberately mirrors Skybox.renderRaw so Ibl.bake can call
    /// either without knowing which it has.
    pub fn renderRaw(
        self: *EnvironmentMap,
        inv_vp: Matrix,
        camera_pos: Vec3,
        pass: PassSignature,
        depth_test: bool,
    ) void {
        const key = PipelineKey{
            .shader = self.shader,
            .layout = .fullscreen,
            .index_type = .u32,
            .indexed = false,
            .pass = pass,
            .primitive = .TRIANGLES,
            .cull = .NONE,
            .blend = .none,
            .depth_test = depth_test,
            .depth_write = false,
        };
        const pip = self.cache.get(key) catch |err| {
            std.log.err("EnvironmentMap: pipeline cache failed: {}", .{err});
            return;
        };

        var params = EnvParams{
            .inv_view_proj = @bitCast(inv_vp),
            .camera_pos = .{ camera_pos.x, camera_pos.y, camera_pos.z, 0 },
            .params = .{ self.intensity, 0, 0, 0 },
        };

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = self.tri.vbuf;

        sg.applyPipeline(pip);
        // The two programs are generated separately, so their slot constants are
        // distinct even where the numbers coincide. Branch on the source rather
        // than assuming they line up.
        switch (self.source) {
            .equirect => {
                bindings.views[shd.VIEW_equirect] = self.view;
                bindings.samplers[shd.SMP_smp] = self.sampler;
                sg.applyBindings(bindings);
                sg.applyUniforms(shd.UB_equirect_params, sg.asRange(&params));
            },
            .cube => {
                bindings.views[shd_cube.VIEW_env_map] = self.view;
                bindings.samplers[shd_cube.SMP_smp] = self.sampler;
                sg.applyBindings(bindings);
                sg.applyUniforms(shd_cube.UB_cube_params, sg.asRange(&params));
            },
        }
        sg.draw(0, 3, 1);
    }
};
