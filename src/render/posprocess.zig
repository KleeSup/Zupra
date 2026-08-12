//! src/render/posprocess.zig
//!
//! Post-process chain: turns the scene's HDR color into the final swapchain
//! image by running an ordered list of fullscreen passes.
//!
//! Two kinds of pass live here:
//!
//!   * BUILT-INS the framework owns — tonemap (always) and anti-aliasing
//!     (swappable via AAMethod).
//!   * USER EFFECTS — any number, added at runtime, each a ShaderProgram plus
//!     uniforms. This is the extension point: chromatic aberration, vignette,
//!     film grain, colour LUTs, outlines, custom stylisation.
//!
//! Effects declare an INJECTION POINT, because colour means different things at
//! different stages and an effect written for one is wrong in the other:
//!
//!     scene_color (HDR, linear)
//!        -> [.hdr effects]        physical light values; bloom, DOF, exposure
//!        -> tonemap
//!        -> [.ldr effects]        display values; aberration, grain, vignette
//!        -> anti-aliasing
//!        -> swapchain
//!
//! This mirrors how the big engines expose it (Unreal's "blendable location",
//! Unity URP's RenderPassEvent, Godot's CompositorEffect callback stages): the
//! user supplies a shader and says WHERE it runs; the engine owns the plumbing.
//!
//! Ping-pong targets are managed here — an effect reads one texture and writes
//! another, and the chain alternates them. The final pass writes straight to the
//! swapchain, so no redundant blit.
//!
//! SHADER CONTRACT — every post-effect fragment shader must declare:
//!     layout(binding=0) uniform texture2D u_input;   // previous stage's colour
//!     layout(binding=1) uniform texture2D u_depth;   // scene depth (opt-in)
//!     layout(binding=0) uniform sampler  u_smp;      // linear, clamped
//!     layout(binding=0) uniform fs_params { ... };   // your parameters
//! and use the standard fullscreen vertex shader (see shaders/chromatic.glsl).

const std = @import("std");
const sg = @import("sokol").gfx;
const zupra = @import("../root.zig");
const pipeline = @import("../graphics/pipeline.zig");
const gfx = @import("../graphics/graphics.zig");
const fb = @import("framebuffer.zig");
const present_mod = @import("present.zig");
const shd_fxaa = @import("shaders").fxaa;
const shd_fxaaq = @import("shaders").fxaa_quality;

const PipelineCache = pipeline.PipelineCache;
const PipelineKey = pipeline.PipelineKey;
const PassSignature = pipeline.PassSignature;
const Framebuffer = fb.Framebuffer;
const Present = present_mod.Present;
const Vertex2D = gfx.Vertex2D;
const Color = zupra.Color;
const Texture = gfx.texture.Texture;
const ShaderProgram = gfx.ShaderProgram;

/// Fixed bind slots every post-effect shader honours. Kept as constants rather
/// than per-effect fields because a uniform contract is what lets the chain
/// treat all effects identically — the same reason SpriteBatch pins its slots.
pub const view_input = 0;
pub const view_depth = 1;
pub const smp_input = 0;

/// Uniform payload cap per effect. Copied into the effect, so callers can build
/// params in a stack local and forget about lifetime (same approach as
/// SpriteBatch.fs_scratch). Raise if an effect genuinely needs more.
pub const max_effect_uniform_bytes = 256;

/// Where an effect runs. Not cosmetic: `.hdr` sees unbounded linear light values
/// (a bright highlight might be 40.0), `.ldr` sees display-referred values in
/// roughly [0,1]. An effect tuned for one produces garbage in the other.
pub const InjectionPoint = enum {
    /// Before tonemap. Physically-meaningful light. Bloom, depth of field,
    /// motion blur, auto-exposure.
    hdr,
    /// After tonemap. Display-referred. Chromatic aberration, vignette, grain,
    /// colour grading LUTs, stylisation.
    ldr,
};

pub const PostEffect = struct {
    shader: ShaderProgram,
    point: InjectionPoint = .ldr,
    enabled: bool = true,
    /// Bind the scene depth texture at `view_depth`. Only request it if the
    /// shader declares u_depth — binding a view a shader doesn't use is a
    /// validation error on some backends.
    wants_depth: bool = false,
    /// For debugging and profiling readouts.
    name: []const u8 = "effect",

    uniform_buf: [max_effect_uniform_bytes]u8 = undefined,
    uniform_len: usize = 0,

    /// Upload parameters for this effect. Pass the generated FsParams struct
    /// from your shader's .glsl.zig. Call whenever values change (per frame is
    /// fine — it's a memcpy).
    pub fn setUniforms(self: *PostEffect, params: anytype) void {
        const bytes = std.mem.asBytes(&params);
        std.debug.assert(bytes.len <= max_effect_uniform_bytes);
        @memcpy(self.uniform_buf[0..bytes.len], bytes);
        self.uniform_len = bytes.len;
    }
};

/// Built-in anti-aliasing. MSAA is deliberately absent: it's hardware
/// multisampling configured on the scene-colour target, a different axis
/// entirely (see SceneRenderer.msaa_samples).
pub const AAMethod = enum {
    none,
    fxaa, // cheap console variant: 9 taps, no edge search
    fxaa_quality, // FXAA 3.11: searches the edge span, keeps texture detail
    taa,
    // smaa,      // needs precomputed area/search LUTs
};

pub const PostChain = struct {
    allocator: std.mem.Allocator,
    cache: *PipelineCache,
    present_: Present,
    aa: AAMethod = .none,

    effects: std.ArrayList(PostEffect) = .empty,

    fxaa_shader: sg.Shader,
    fxaaq_shader: sg.Shader,
    sampler: sg.Sampler,
    fullscreen_vbuf: sg.Buffer,

    /// Min local luma contrast to treat as an edge. Lower = more edges caught
    /// (softer), higher = only strong edges (sharper, more aliasing).
    edge_threshold: f32 = 0.166,
    /// How much sub-pixel detail gets low-passed. 0 = crisp but shimmery on thin
    /// geometry, 1 = smooth but softer.
    subpix_quality: f32 = 0.75,

    /// LDR ping-pong. `ldr_a` always receives the tonemap output.
    ldr_a: Framebuffer,
    ldr_b: Framebuffer,
    /// HDR ping-pong, allocated only when an .hdr effect exists — most projects
    /// never have one, and RGBA16F targets aren't cheap.
    hdr_a: ?Framebuffer = null,
    hdr_b: ?Framebuffer = null,

    width: u32,
    height: u32,
    clear_color: Color,

    pub fn init(
        allocator: std.mem.Allocator,
        cache: *PipelineCache,
        width: u32,
        height: u32,
        clear_color: Color,
    ) PostChain {
        const fxaa_shader = sg.makeShader(shd_fxaa.fxaaShaderDesc(sg.queryBackend()));
        const fxaaq_shader = sg.makeShader(shd_fxaaq.fxaaQualityShaderDesc(sg.queryBackend()));
        const sampler = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        // Fullscreen triangle, backend-correct UV origin.
        const top_left = sg.queryFeatures().origin_top_left;
        const clip = [3][2]f32{ .{ -1, -1 }, .{ 3, -1 }, .{ -1, 3 } };
        var verts: [3]Vertex2D = undefined;
        for (clip, 0..) |p, i| {
            const u = 0.5 + 0.5 * p[0];
            const v = if (top_left) 0.5 - 0.5 * p[1] else 0.5 + 0.5 * p[1];
            verts[i] = .{ .pos = p, .uv = .{ u, v }, .color = 0xFFFFFFFF };
        }
        const vbuf = sg.makeBuffer(.{ .data = sg.asRange(&verts) });

        return PostChain{
            .allocator = allocator,
            .cache = cache,
            .present_ = Present.init(cache),
            .fxaa_shader = fxaa_shader,
            .fxaaq_shader = fxaaq_shader,
            .sampler = sampler,
            .fullscreen_vbuf = vbuf,
            .ldr_a = Framebuffer.init(.{ .width = width, .height = height, .color_format = .RGBA8 }),
            .ldr_b = Framebuffer.init(.{ .width = width, .height = height, .color_format = .RGBA8 }),
            .width = width,
            .height = height,
            .clear_color = clear_color,
        };
    }

    pub fn deinit(self: *PostChain) void {
        self.effects.deinit(self.allocator);
        self.ldr_a.deinit();
        self.ldr_b.deinit();
        if (self.hdr_a) |*f| f.deinit();
        if (self.hdr_b) |*f| f.deinit();
        self.present_.deinit();
        sg.destroyBuffer(self.fullscreen_vbuf);
        sg.destroySampler(self.sampler);
        sg.destroyShader(self.fxaa_shader);
        sg.destroyShader(self.fxaaq_shader);
    }

    pub fn resize(self: *PostChain, width: u32, height: u32) void {
        if (width == self.width and height == self.height) return;
        self.ldr_a.deinit();
        self.ldr_b.deinit();
        self.ldr_a = Framebuffer.init(.{ .width = width, .height = height, .color_format = .RGBA8 });
        self.ldr_b = Framebuffer.init(.{ .width = width, .height = height, .color_format = .RGBA8 });
        if (self.hdr_a) |*f| {
            f.deinit();
            self.hdr_a = Framebuffer.init(.{ .width = width, .height = height, .color_format = .RGBA16F });
        }
        if (self.hdr_b) |*f| {
            f.deinit();
            self.hdr_b = Framebuffer.init(.{ .width = width, .height = height, .color_format = .RGBA16F });
        }
        self.width = width;
        self.height = height;
    }

    // ---------------------------------------------------------------------
    //  User-facing effect management
    // ---------------------------------------------------------------------

    /// Append an effect. Returns a pointer so the caller can toggle `enabled`
    /// or push new uniforms each frame:
    ///
    ///     const chroma = try scene.post.addEffect(.{
    ///         .shader = ShaderProgram.init(shaders.chromatic.chromaticShaderDesc, .{
    ///             .layout = .fullscreen,
    ///             .slots = .{ .fs_params = shaders.chromatic.UB_fs_params },
    ///         }),
    ///         .point = .ldr,
    ///         .name = "chromatic aberration",
    ///     });
    ///     chroma.setUniforms(shaders.chromatic.FsParams{ .params = .{ 0.005, 0, 0, 0 } });
    ///
    /// Effects run in insertion order within their injection point.
    pub fn addEffect(self: *PostChain, effect: PostEffect) !*PostEffect {
        try self.effects.append(self.allocator, effect);
        return &self.effects.items[self.effects.items.len - 1];
    }

    /// Look up an effect by name (linear scan — the list is tiny).
    pub fn findEffect(self: *PostChain, name: []const u8) ?*PostEffect {
        for (self.effects.items) |*e| {
            if (std.mem.eql(u8, e.name, name)) return e;
        }
        return null;
    }

    pub fn clearEffects(self: *PostChain) void {
        self.effects.clearRetainingCapacity();
    }

    fn countEnabled(self: PostChain, point: InjectionPoint) usize {
        var n: usize = 0;
        for (self.effects.items) |e| {
            if (e.enabled and e.point == point) n += 1;
        }
        return n;
    }

    fn ensureHdrTargets(self: *PostChain) void {
        if (self.hdr_a != null) return;
        self.hdr_a = Framebuffer.init(.{ .width = self.width, .height = self.height, .color_format = .RGBA16F });
        self.hdr_b = Framebuffer.init(.{ .width = self.width, .height = self.height, .color_format = .RGBA16F });
    }

    // ---------------------------------------------------------------------
    //  Frame
    // ---------------------------------------------------------------------

    /// Produce the final swapchain image. `depth` is the scene depth texture,
    /// needed only by effects with `wants_depth`; pass null if unavailable.
    pub fn present(self: *PostChain, scene_color: Texture, depth: ?Texture) void {
        // --- HDR stage: effects that need linear light values ---
        var hdr_src = scene_color;
        var hdr_left = self.countEnabled(.hdr);
        if (hdr_left > 0) {
            self.ensureHdrTargets();
            var slot: usize = 0;
            for (self.effects.items) |*e| {
                if (!e.enabled or e.point != .hdr) continue;
                const target: *Framebuffer = if (slot % 2 == 0) &self.hdr_a.? else &self.hdr_b.?;
                zupra.beginDrawingFramebufferClear(target.*, self.clear_color);
                self.runEffect(e, hdr_src, depth, target.passSignature());
                zupra.endDrawing();
                hdr_src = target.asTexture();
                slot += 1;
                hdr_left -= 1;
            }
        }

        // --- Tonemap ---
        // If nothing follows, tonemap straight to the swapchain (no intermediate).
        const ldr_count = self.countEnabled(.ldr);
        if (ldr_count == 0 and !self.postAaRuns()) {
            zupra.beginDrawingClear(self.clear_color);
            self.present_.render(hdr_src, PassSignature.swapchainPass());
            zupra.endDrawing();
            return;
        }

        zupra.beginDrawingFramebufferClear(self.ldr_a, self.clear_color);
        self.present_.render(hdr_src, self.ldr_a.passSignature());
        zupra.endDrawing();
        var ldr_src = self.ldr_a.asTexture();

        // --- LDR stage: display-referred effects ---
        var ldr_left = ldr_count;
        var slot: usize = 0;
        for (self.effects.items) |*e| {
            if (!e.enabled or e.point != .ldr) continue;
            ldr_left -= 1;

            // The last thing in the whole chain writes directly to the
            // swapchain — no pointless blit through an intermediate.
            if (ldr_left == 0 and !self.postAaRuns()) {
                zupra.beginDrawingClear(self.clear_color);
                self.runEffect(e, ldr_src, depth, PassSignature.swapchainPass());
                zupra.endDrawing();
                return;
            }

            // Ping-pong: ldr_a holds the tonemap output, so the first effect
            // writes to ldr_b, the next back to ldr_a, and so on.
            const target: *Framebuffer = if (slot % 2 == 0) &self.ldr_b else &self.ldr_a;
            zupra.beginDrawingFramebufferClear(target.*, self.clear_color);
            self.runEffect(e, ldr_src, depth, target.passSignature());
            zupra.endDrawing();
            ldr_src = target.asTexture();
            slot += 1;
        }

        // --- Anti-aliasing (always last: it wants the finished image) ---
        if (self.postAaRuns()) {
            zupra.beginDrawingClear(self.clear_color);
            switch (self.aa) {
                .none, .taa => unreachable, // postAaRuns() excluded these
                .fxaa => self.fxaa(ldr_src, PassSignature.swapchainPass()),
                .fxaa_quality => self.fxaaQuality(ldr_src, PassSignature.swapchainPass()),
            }
            zupra.endDrawing();
        }
    }

    /// Whether the post chain itself runs an AA filter at the end. TAA resolves
    /// upstream in SceneRenderer, so as far as this chain is concerned it is the
    /// same as no filter, and the last pass can write straight to the swapchain.
    fn postAaRuns(self: PostChain) bool {
        return switch (self.aa) {
            .none, .taa => false,
            .fxaa, .fxaa_quality => true,
        };
    }

    /// Run one user effect as a fullscreen pass. Everything an effect needs is
    /// bound by contract, so the chain never has to know what the shader does.
    fn runEffect(self: *PostChain, e: *PostEffect, input: Texture, depth: ?Texture, pass: PassSignature) void {
        const key = PipelineKey{
            .shader = e.shader.handle,
            .layout = .fullscreen,
            .index_type = .u16, // unused
            .indexed = false,
            .pass = pass,
            .primitive = .TRIANGLES,
            .cull = .NONE,
            .blend = .none,
            .depth_test = false,
            .depth_write = false,
        };
        const pip = self.cache.get(key) catch |err| {
            std.log.err("PostChain: pipeline failed for effect '{s}': {}", .{ e.name, err });
            return;
        };

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = self.fullscreen_vbuf;
        bindings.views[view_input] = input.view;
        bindings.samplers[smp_input] = self.sampler;
        if (e.wants_depth) {
            if (depth) |d| {
                bindings.views[view_depth] = d.view;
            } else {
                std.log.warn("PostChain: effect '{s}' wants depth but none was provided", .{e.name});
                return;
            }
        }

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        if (e.uniform_len > 0) {
            const slot_idx = e.shader.slots.fs_params orelse 0;
            sg.applyUniforms(slot_idx, sg.asRange(e.uniform_buf[0..e.uniform_len]));
        } else {
            zupra.log.err("PostChain: effect '{s}' has no uniforms set. Make sure to call setUniforms() before the first frame!", .{e.name});
            return;
        }
        sg.draw(0, 3, 1);
    }

    // ---------------------------------------------------------------------
    //  Built-in AA
    // ---------------------------------------------------------------------

    fn fxaa(self: *PostChain, input: Texture, pass: PassSignature) void {
        const key = PipelineKey{
            .shader = self.fxaa_shader,
            .layout = .fullscreen,
            .index_type = .u16,
            .indexed = false,
            .pass = pass,
            .primitive = .TRIANGLES,
            .cull = .NONE,
            .blend = .none,
            .depth_test = false,
            .depth_write = false,
        };
        const pip = self.cache.get(key) catch return;

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = self.fullscreen_vbuf;
        bindings.views[shd_fxaa.VIEW_scene_ldr] = input.view;
        bindings.samplers[shd_fxaa.SMP_smp] = self.sampler;

        var fs = shd_fxaa.FsParams{ .inv_resolution = .{
            1.0 / @as(f32, @floatFromInt(self.width)),
            1.0 / @as(f32, @floatFromInt(self.height)),
            0,
            0,
        } };

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(shd_fxaa.UB_fs_params, sg.asRange(&fs));
        sg.draw(0, 3, 1);
    }

    fn fxaaQuality(self: *PostChain, input: Texture, pass: PassSignature) void {
        const key = PipelineKey{
            .shader = self.fxaaq_shader,
            .layout = .fullscreen,
            .index_type = .u16,
            .indexed = false,
            .pass = pass,
            .primitive = .TRIANGLES,
            .cull = .NONE,
            .blend = .none,
            .depth_test = false,
            .depth_write = false,
        };
        const pip = self.cache.get(key) catch return;

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = self.fullscreen_vbuf;
        bindings.views[shd_fxaaq.VIEW_scene_ldr] = input.view;
        bindings.samplers[shd_fxaaq.SMP_smp] = self.sampler;

        var fs = shd_fxaaq.FsParams{ .inv_resolution = .{
            1.0 / @as(f32, @floatFromInt(self.width)),
            1.0 / @as(f32, @floatFromInt(self.height)),
            self.edge_threshold,
            self.subpix_quality,
        } };

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(shd_fxaaq.UB_fs_params, sg.asRange(&fs));
        sg.draw(0, 3, 1);
    }
};
