//! src/render/material.zig
//!
//! PBR material, glTF metallic-roughness model. Architecture:
//!
//!   * Scalar FACTORS are flat fields (always present which is still very cheap).
//!   * Texture MAPS are optional: a null map resolves to a framework default
//!     1x1 texture (white / flat-normal), so the shader always has every
//!     sampler bound.
//!   * Final value = factor * sampled_map. With a white default this gives the
//!     factor directly when no texture is set.
//!   * The material also drives PIPELINE STATE (cull from double_sided, blend
//!     from alpha_mode), because in the framework the material should fully
//!     describes how to render the surface.
//!   * A MapSlot enum + map() resolver let the renderer bind maps in a uniform
//!     loop and the shader declare them at matching slots (one contract).
//!

const std = @import("std");
const sg = @import("sokol").gfx;
const tex = @import("../graphics/texture.zig");
const pipeline = @import("../graphics/pipeline.zig");
const zupra = @import("../root.zig");
const shader = @import("../graphics/shader.zig");

const Texture = tex.Texture;
const Color = zupra.Color;
const BlendMode = pipeline.BlendMode;
const ShaderProgram = shader.ShaderProgram;

// --- framework default textures (lazy 1x1, created after sg.setup) ---

var white_tex: ?Texture = null;
var black_tex: ?Texture = null;
var flat_normal_tex: ?Texture = null;

pub fn whiteTexture() Texture {
    if (white_tex == null) {
        const px = [_]u8{ 255, 255, 255, 255 };
        white_tex = Texture.initRaw(&px, 1, 1);
    }
    return white_tex.?;
}
pub fn blackTexture() Texture {
    if (black_tex == null) {
        const px = [_]u8{ 0, 0, 0, 255 };
        black_tex = Texture.initRaw(&px, 1, 1);
    }
    return black_tex.?;
}
/// Flat tangent-space normal (0,0,1) encoded as (0.5,0.5,1.0): no perturbation.
pub fn flatNormalTexture() Texture {
    if (flat_normal_tex == null) {
        const px = [_]u8{ 128, 128, 255, 255 };
        flat_normal_tex = Texture.initRaw(&px, 1, 1);
    }
    return flat_normal_tex.?;
}

pub fn deinitDefaults() void {
    if (white_tex) |t| {
        t.deinit();
        white_tex = null;
    }
    if (black_tex) |t| {
        t.deinit();
        black_tex = null;
    }
    if (flat_normal_tex) |t| {
        t.deinit();
        flat_normal_tex = null;
    }
}

// --- material ---

pub const max_material_uniform_bytes = 256;

/// How a surface is shaded. Lets you opt out of PBR for stylized/low-poly
/// work without losing access to it elsewhere.
///   .pbr     — full Cook-Torrance + IBL (default; realistic).
///   .lambert — diffuse N·L + flat ambient; no specular/metallic/IBL (simple lit).
///   .unlit   — flat albedo, no lighting (UI-in-world, stylized, debug).
pub const ShadingModel = enum(u8) {
    pbr,
    lambert,
    unlit,
};

pub const AlphaMode = enum(u8) {
    /// Fully opaque; alpha ignored.
    opaque_,
    /// Alpha-tested: fragments below alpha_cutoff are discarded (foliage, etc).
    mask,
    /// Alpha-blended (transparent).
    blend,
};

/// Texture channels, in binding-slot order. The shader declares textures at
/// these same slot indices; the renderer binds map(slot) for each.
pub const MapSlot = enum(u8) {
    base_color = 0,
    normal = 1,
    metallic_roughness = 2, // glTF packing: G = roughness, B = metallic
    occlusion = 3, // R channel
    emissive = 4,
};
pub const map_slot_count = @typeInfo(MapSlot).@"enum".fields.len;

/// Per-map UV transform (glTF KHR_texture_transform).
///
/// The extension composes translation * rotation * scale. Rotation is baked
/// into a 2x2 at LOAD time, so the shader never evaluates sin/cos, so it just
/// does one 2x2 multiply plus an add:
///
///     uv' = M * uv + offset,  M = [ m[0] m[1] ]
///                                 [ m[2] m[3] ]
///
/// Identity by default, so procedural materials cost nothing.
pub const UvXform = extern struct {
    /// Packed 2x2 (rotation * scale), row-major: m00, m01, m10, m11.
    m: [4]f32 = .{ 1, 0, 0, 1 },
    offset: [2]f32 = .{ 0, 0 },
};

pub const Material = struct {
    // --- factors ---
    base_color: Color = .{ .r = 1, .g = 1, .b = 1, .a = 1 },
    metallic: f32 = 0.0,
    roughness: f32 = 0.5,
    emissive: Color = .{ .r = 0, .g = 0, .b = 0, .a = 1 }, // rgb used
    emissive_strength: f32 = 1.0,
    normal_scale: f32 = 1.0,

    /// Important for GL vs DX context where the green channel for the normal needs to be inverted for DX. Default is GL standard which does not need a flip.
    /// The flip is encoded as the sign of the normal_scale.
    normal_flip_y: bool = false,
    occlusion_strength: f32 = 1.0,
    alpha_cutoff: f32 = 0.5,
    uv_scale: [2]f32 = .{ 1, 1 },

    // --- maps (null -> default texture; final = factor * sample) ---
    base_color_map: ?Texture = null,
    metallic_roughness_map: ?Texture = null,
    normal_map: ?Texture = null,
    occlusion_map: ?Texture = null,
    emissive_map: ?Texture = null,

    // --- surface/state flags ---
    alpha_mode: AlphaMode = .opaque_,
    double_sided: bool = false,
    cull_override: ?sg.CullMode = null,
    sampler: ?sg.Sampler = null,

    /// Bits follow MapIndex (0 base_color, 1 normal, 2 metallic_roughness,
    /// 3 occlusion, 4 emissive). glTF allows a per-map UV set; occlusion on
    /// UV1 is common (baked AO uses its own unwrap). Then UvXform carries only the KHR transform.
    uv_set: u8 = 0,
    /// Per-map KHR_texture_transform. Identity unless the asset sets it.
    uv_xforms: [map_slot_count]UvXform = @splat(.{}),

    shading: ShadingModel = .pbr,

    shader: ?ShaderProgram = null,
    shader_writes_gbuffer: bool = false,
    shader_uniforms: [max_material_uniform_bytes]u8 = undefined,
    shader_uniform_len: usize = 0,

    pub fn usesUv1(self: Material, comptime map_: comptime_int) bool {
        return (self.uv_set & (@as(u8, 1) << map_)) != 0;
    }

    /// Resolve a channel to its texture, falling back to the framework default
    /// so the shader always has a bound sampler.
    pub fn map(self: Material, slot: MapSlot) Texture {
        return switch (slot) {
            .base_color => self.base_color_map orelse whiteTexture(),
            .metallic_roughness => self.metallic_roughness_map orelse whiteTexture(),
            .normal => self.normal_map orelse flatNormalTexture(),
            .occlusion => self.occlusion_map orelse whiteTexture(),
            .emissive => self.emissive_map orelse whiteTexture(),
        };
    }

    // --- pipeline state derived from the material ---

    pub fn cullMode(self: Material) sg.CullMode {
        if (self.cull_override) |c| return c;
        return if (self.double_sided) .NONE else .BACK;
    }

    pub fn blendMode(self: Material) BlendMode {
        return switch (self.alpha_mode) {
            .blend => .alpha,
            else => .none, // opaque and mask both write opaque, then mask discards in shader
        };
    }

    /// Parameters shared by every built-in material pass that can alpha-test a
    /// glTF base-colour texture. `x` is the glTF cutoff and `y` is 1 only for
    /// alpha-mask materials. Keeping the mode flag explicit matters because a
    /// cutoff of 0 is valid and must not be mistaken for "testing disabled".
    pub fn alphaTestParams(self: Material) [4]f32 {
        return .{ self.alpha_cutoff, if (self.alpha_mode == .mask) 1.0 else 0.0, 0, 0 };
    }

    // --- shader ---

    /// Upload parameters for the custom shader's fs_params block. Pass the
    /// generated FsParams struct from your shader's .glsl.zig.
    pub fn setShaderUniforms(self: *Material, params: anytype) void {
        const bytes = std.mem.asBytes(&params);
        std.debug.assert(bytes.len <= max_material_uniform_bytes);
        @memcpy(self.shader_uniforms[0..bytes.len], bytes);
        self.shader_uniform_len = bytes.len;
    }

    /// True when this material must take the forward path regardless of mode.
    pub fn requiresForward(self: Material) bool {
        return self.shader != null and !self.shader_writes_gbuffer;
    }
};
