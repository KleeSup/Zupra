const Light = @import("light.zig").Light;
const Color = @import("../root.zig").Color;

pub const Environment = struct {
    lights: []const Light = &.{}, // borrowed slice; caller owns storage
    ambient: Color = .{ .r = 0.1, .g = 0.1, .b = 0.12, .a = 1 },
    // later: skybox: ?Cubemap, irradiance/prefilter/brdf_lut for IBL

    pub fn addLight(self: *Environment, light: Light) void {
        // caller is responsible for ensuring capacity of `lights` slice
        self.lights[self.lights.len] = light;
        self.lights = self.lights[0 .. self.lights.len + 1];
    }

    pub fn clearLights(self: *Environment) void {
        self.lights = self.lights[0..0];
    }
};
