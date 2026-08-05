//! src/render/environment.zig
//!
//! Scene lighting state, consumed by both the forward (MeshRenderer/ModelBatch)
//! and deferred (DeferredRenderer) paths. It is solely data, not a renderer or a
//! shader; each renderer reads it and applies its own math, so one Environment
//! can drive forward, deferred, and the skybox/transparent passes.
//!
//! Lights are CLUSTERED (see lighting.zig / cluster.zig): directional lights go
//! in a small uniform block, point/spot lights into a texture that the shader
//! reads per-froxel. So there is no 16-light cap any more - addLight returns a
//! stable handle and hundreds of punctual lights are fine.
//!
//! Environment now owns GPU resources (via LightingFrame), so it must be passed
//! BY POINTER, not by value.

const std = @import("std");
const light_mod = @import("light.zig");
const lighting_mod = @import("lighting.zig");
const cluster_mod = @import("cluster.zig");
const Light = light_mod.Light;
const LightHandle = light_mod.LightHandle;
const LightingFrame = lighting_mod.LightingFrame;
const ClusterOptions = cluster_mod.ClusterOptions;
const Color = @import("../root.zig").Color;

pub const Environment = struct {
    lighting: LightingFrame,
    ambient: Color = .{ .r = 0.1, .g = 0.1, .b = 0.12, .a = 1 },

    pub fn init(allocator: std.mem.Allocator) Environment {
        return .{ .lighting = LightingFrame.init(allocator, .{}) };
    }

    /// With a custom froxel grid (coarser for mobile, finer for dense scenes).
    pub fn initWith(allocator: std.mem.Allocator, opts: ClusterOptions) Environment {
        return .{ .lighting = LightingFrame.init(allocator, opts) };
    }

    pub fn deinit(self: *Environment) void {
        self.lighting.deinit();
    }

    /// Append a light; returns a stable handle. No cap for point/spot (they're
    /// clustered); directional lights are limited to light.zig max_directional.
    pub fn addLight(self: *Environment, light: Light) !LightHandle {
        return self.lighting.addLight(light);
    }

    pub fn removeLight(self: *Environment, handle: LightHandle) void {
        self.lighting.removeLight(handle);
    }

    /// Mutable access to move/recolour a light after creation.
    pub fn getLight(self: *Environment, handle: LightHandle) ?*Light {
        return self.lighting.getLight(handle);
    }

    pub fn clearLights(self: *Environment) void {
        self.lighting.clearLights();
    }

    pub fn lightCount(self: *const Environment) usize {
        return self.lighting.lightCount();
    }
};
