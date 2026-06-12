//! src/render/light.zig
//!
//! Lights + the shared CPU->GPU light uniform packing used by BOTH the forward
//! (mesh.glsl) and deferred (deferred_lighting.glsl) paths.
//!
//! Light types: directional, point, spot. The uniform layout is now final —
//! position, direction, range, and spot cone all have a home, so adding a light
//! type never re-churns LightParams.
//!
//! Convention: light.direction is the TRAVEL direction (where the light points,
//! e.g. a sun pointing down = {0,-1,0}; a spot's cone axis). The shaders derive
//! the direction-to-light from it (directional: negate; point/spot: position).

const math = @import("../math.zig");
const Vec3 = math.Vec3;
const DEG_TO_RAD = math.DEG_TO_RAD;
const Color = @import("../root.zig").Color;

pub const LightType = enum(u32) { directional = 0, point = 1, spot = 2 };

pub const Light = struct {
    type: LightType = .directional,
    /// Travel direction (sun pointing down = {0,-1,0}; spot cone axis).
    direction: Vec3 = .{ .x = -0.4, .y = -1.0, .z = -0.3 },
    /// World position (point/spot only).
    position: Vec3 = .{ .x = 0, .y = 0, .z = 0 },
    color: Color = .{ .r = 1, .g = 1, .b = 1, .a = 1 },
    intensity: f32 = 3.0,
    /// Falloff radius for point/spot (light is ~0 beyond this).
    range: f32 = 20.0,
    /// Spot cone half-angles in degrees: full intensity within inner, fading to
    /// zero at outer.
    spot_inner_deg: f32 = 20.0,
    spot_outer_deg: f32 = 30.0,

    pub fn directional(dir: Vec3, color: Color, intensity: f32) Light {
        return .{ .type = .directional, .direction = dir, .color = color, .intensity = intensity };
    }

    /// Point lights use inverse-square falloff, so they generally need a higher
    /// intensity than directional lights to read at a distance.
    pub fn point(position: Vec3, color: Color, intensity: f32, range: f32) Light {
        return .{ .type = .point, .position = position, .color = color, .intensity = intensity, .range = range };
    }

    pub fn spot(position: Vec3, direction: Vec3, color: Color, intensity: f32, range: f32, inner_deg: f32, outer_deg: f32) Light {
        return .{
            .type = .spot,
            .position = position,
            .direction = direction,
            .color = color,
            .intensity = intensity,
            .range = range,
            .spot_inner_deg = inner_deg,
            .spot_outer_deg = outer_deg,
        };
    }
};

pub const MAX_LIGHTS = 16;

// std140-matching uniform block. vec4 arrays pack tightly at 16 bytes each.
// Field order is a CONTRACT with both shaders' uniform blocks.
pub const LightParams = extern struct {
    camera_pos: [4]f32,
    ambient_count: [4]f32, // rgb ambient, w = light count
    light_pos: [MAX_LIGHTS][4]f32, // xyz world position, w = type
    light_dir: [MAX_LIGHTS][4]f32, // xyz travel direction, w = range
    light_color: [MAX_LIGHTS][4]f32, // rgb, w = intensity
    light_spot: [MAX_LIGHTS][4]f32, // x = cos(inner), y = cos(outer)
};

pub fn packLightParams(lights: []const Light, ambient: Color, camera_pos: Vec3) LightParams {
    var p: LightParams = undefined;
    p.camera_pos = .{ camera_pos.x, camera_pos.y, camera_pos.z, 0 };
    const n = @min(lights.len, MAX_LIGHTS);
    p.ambient_count = .{ ambient.r, ambient.g, ambient.b, @floatFromInt(n) };

    for (0..MAX_LIGHTS) |i| {
        p.light_pos[i] = .{ 0, 0, 0, 0 };
        p.light_dir[i] = .{ 0, 0, 0, 0 };
        p.light_color[i] = .{ 0, 0, 0, 0 };
        p.light_spot[i] = .{ 0, 0, 0, 0 };
    }
    for (lights[0..n], 0..) |l, i| {
        const t: f32 = @floatFromInt(@intFromEnum(l.type));
        p.light_pos[i] = .{ l.position.x, l.position.y, l.position.z, t };
        p.light_dir[i] = .{ l.direction.x, l.direction.y, l.direction.z, l.range };
        p.light_color[i] = .{ l.color.r, l.color.g, l.color.b, l.intensity };
        p.light_spot[i] = .{ @cos(l.spot_inner_deg * DEG_TO_RAD), @cos(l.spot_outer_deg * DEG_TO_RAD), 0, 0 };
    }
    return p;
}
