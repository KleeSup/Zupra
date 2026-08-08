pub const sprite = @import("sprite.glsl.zig");
pub const debug = @import("debug.glsl.zig");
pub const sdf = @import("text_sdf.glsl.zig");
pub const mesh = @import("mesh.glsl.zig");

pub const gbuffer = @import("gbuffer.glsl.zig");
pub const deferred_lighting = @import("deferred_lighting.glsl.zig");

pub const tonemap = @import("tonemap.glsl.zig");
pub const sky = @import("sky.glsl.zig");
pub const irradiance = @import("irradiance.glsl.zig");
pub const prefilter = @import("prefilter.glsl.zig");
pub const sky_equirect = @import("sky_equirect.glsl.zig");
pub const sky_cube = @import("sky_cube.glsl.zig");

pub const lambert = @import("lambert.glsl.zig");
pub const unlit = @import("unlit.glsl.zig");

pub const fxaa = @import("fxaa.glsl.zig");
pub const fxaa_quality = @import("fxaa_quality.glsl.zig");

pub const shadow_depth = @import("shadow_depth.glsl.zig");
pub const shadow_depth_instanced = @import("shadow_depth_instanced.glsl.zig");
pub const depth_prepass = @import("depth_prepass.glsl.zig");

pub const test_hologram = @import("hologram.glsl.zig");
