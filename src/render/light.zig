const Vec3 = @import("../math.zig").Vec3;
const Color = @import("../root.zig").Color;

// =====================================================================
// Lights
// =====================================================================

pub const LightType = enum(u32) { directional = 0, point = 1, spot = 2 };

pub const Light = struct {
    type: LightType = .directional,
    /// Travel direction (sun pointing down = {0,-1,0}). Negated to dir-to-light.
    direction: Vec3 = .{ .x = -0.4, .y = -1.0, .z = -0.3 },
    position: Vec3 = .{ .x = 0, .y = 0, .z = 0 },
    color: Color = .{ .r = 1, .g = 1, .b = 1, .a = 1 },
    intensity: f32 = 3.0,
    range: f32 = 20.0,

    pub fn directional(dir: Vec3, color: Color, intensity: f32) Light {
        return .{ .type = .directional, .direction = dir, .color = color, .intensity = intensity };
    }
};

// =====================================================================
// Lighting pass (fullscreen PBR)
// =====================================================================

pub const MAX_LIGHTS = 16;

// std140-matching uniform block (vec4 arrays pack tightly at 16 bytes each).
const LightParams = extern struct {
    camera_pos: [4]f32,
    ambient_count: [4]f32,
    light_dir: [MAX_LIGHTS][4]f32,
    light_color: [MAX_LIGHTS][4]f32,
};
