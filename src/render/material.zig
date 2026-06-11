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

const Texture = tex.Texture;
const Color = zupra.Color;
const BlendMode = pipeline.BlendMode;

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
    metallic_roughness = 1, // glTF packing: G = roughness, B = metallic
    normal = 2,
    occlusion = 3, // R channel
    emissive = 4,
};
pub const map_slot_count = @typeInfo(MapSlot).@"enum".fields.len;

pub const Material = struct {
    // --- factors (always present) ---
    base_color: Color = .{ .r = 1, .g = 1, .b = 1, .a = 1 },
    metallic: f32 = 0.0,
    roughness: f32 = 0.5,
    emissive: Color = .{ .r = 0, .g = 0, .b = 0, .a = 1 }, // rgb used
    normal_scale: f32 = 1.0,
    occlusion_strength: f32 = 1.0,
    alpha_cutoff: f32 = 0.5,

    // --- maps (null -> default texture; final = factor * sample) ---
    base_color_map: ?Texture = null,
    metallic_roughness_map: ?Texture = null,
    normal_map: ?Texture = null,
    occlusion_map: ?Texture = null,
    emissive_map: ?Texture = null,

    // --- surface/state flags ---
    alpha_mode: AlphaMode = .opaque_,
    double_sided: bool = false,

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
        return if (self.double_sided) .NONE else .BACK;
    }

    pub fn blendMode(self: Material) BlendMode {
        return switch (self.alpha_mode) {
            .blend => .alpha,
            else => .none, // opaque + mask both write opaque; mask discards in shader
        };
    }
};
