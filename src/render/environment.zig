//! src/render/environment.zig
//!
//! Scene lighting state, consumed by both the forward (MeshRenderer/ModelBatch)
//! and deferred (DeferredRenderer) paths. It is soley data, not a renderer or a shader, then
//! each renderer reads it and applies its own math, so one Environment can
//! drive forward, deferred, and the skybox/transparent passes.
//!
//! Owns a fixed light buffer (capped at MAX_LIGHTS, the same cap the GPU uniform
//! uses) so addLight/clearLights work without heap and without dangling-slice
//! risk. Read the active lights via lights().

const light_mod = @import("light.zig");
const Light = light_mod.Light;
const Color = @import("../root.zig").Color;

pub const Environment = struct {
    lights_buf: [light_mod.MAX_LIGHTS]Light = undefined,
    light_count: usize = 0,
    ambient: Color = .{ .r = 0.1, .g = 0.1, .b = 0.12, .a = 1 },

    /// Append a light; silently ignored past MAX_LIGHTS.
    pub fn addLight(self: *Environment, light: Light) void {
        if (self.light_count >= self.lights_buf.len) return;
        self.lights_buf[self.light_count] = light;
        self.light_count += 1;
    }

    pub fn clearLights(self: *Environment) void {
        self.light_count = 0;
    }

    /// The active lights, for packing into the GPU uniform.
    pub fn lights(self: *const Environment) []const Light {
        return self.lights_buf[0..self.light_count];
    }
};
