//! src/render/fpscontroller.zig
//!
//! First-person camera controller: WASD/QE movement + mouse look, driving a
//! Camera3D. Framework-style (not engine): it holds its own position/orientation
//! and writes into a Camera3D you own — it doesn't hide the camera.
//!
//! Wiring (three touch points):
//!   1. handleEvent(ev) — forward every sokol app event here (needs the app to
//!      expose an event callback; see note at bottom).
//!   2. update(dt)      — call once per frame with delta seconds.
//!   3. applyTo(&camera)— call after update to set the camera's pose.
//!
//! Mouse look needs a locked pointer: call setMouseLocked(true) (e.g. on click)
//! so relative mouse deltas arrive.

const std = @import("std");
const sapp = @import("sokol").app;
const math = @import("../math.zig");
const Camera3D = @import("camera3d.zig").Camera3D;

const Vec3 = math.Vec3;

pub const FirstPersonController = struct {
    position: Vec3,
    yaw: f32 = 0, // radians, around +Y; 0 looks toward +Z
    pitch: f32 = 0, // radians, up(+)/down(-)

    move_speed: f32 = 5.0, // units/second
    sprint_mult: f32 = 3.0, // hold shift
    look_sensitivity: f32 = 0.0022, // radians per mouse pixel

    // held movement keys
    fwd: bool = false,
    back: bool = false,
    left: bool = false,
    right: bool = false,
    up: bool = false,
    down: bool = false,
    sprint: bool = false,

    mouse_locked: bool = false,

    const pitch_limit: f32 = std.math.pi * 0.5 - 0.01; // avoid gimbal flip at poles

    pub fn init(position: Vec3) FirstPersonController {
        return .{ .position = position };
    }

    /// Unit forward direction from yaw/pitch (left-handed, Y-up).
    pub fn forward(self: FirstPersonController) Vec3 {
        const cp = @cos(self.pitch);
        return .{
            .x = @sin(self.yaw) * cp,
            .y = @sin(self.pitch),
            .z = @cos(self.yaw) * cp,
        };
    }

    /// Horizontal right vector (for strafing), independent of pitch.
    pub fn rightVec(self: FirstPersonController) Vec3 {
        return .{ .x = @cos(self.yaw), .y = 0, .z = -@sin(self.yaw) };
    }

    pub fn setMouseLocked(self: *FirstPersonController, locked: bool) void {
        self.mouse_locked = locked;
        sapp.lockMouse(locked);
    }

    /// Feed sokol app events. Handles key up/down (movement) and mouse motion
    /// (look, while locked).
    pub fn handleEvent(self: *FirstPersonController, ev: *const sapp.Event) void {
        switch (ev.type) {
            .MOUSE_MOVE => {
                if (self.mouse_locked) {
                    self.yaw += ev.mouse_dx * self.look_sensitivity;
                    self.pitch -= ev.mouse_dy * self.look_sensitivity;
                    self.pitch = std.math.clamp(self.pitch, -pitch_limit, pitch_limit);
                }
            },
            .MOUSE_DOWN => self.setMouseLocked(true),
            .KEY_DOWN, .KEY_UP => {
                const down = ev.type == .KEY_DOWN;
                switch (ev.key_code) {
                    .W => self.fwd = down,
                    .S => self.back = down,
                    .A => self.left = down,
                    .D => self.right = down,
                    .E, .SPACE => self.up = down,
                    .Q, .LEFT_CONTROL => self.down = down,
                    .LEFT_SHIFT => self.sprint = down,
                    .ESCAPE => if (down) self.setMouseLocked(false),
                    else => {},
                }
            },
            else => {},
        }
    }

    /// Integrate held-key movement over dt seconds.
    pub fn update(self: *FirstPersonController, dt: f32) void {
        const f = self.forward();
        const r = self.rightVec();

        var move = Vec3{ .x = 0, .y = 0, .z = 0 };
        if (self.fwd) move = add(move, f);
        if (self.back) move = add(move, scale(f, -1));
        if (self.right) move = add(move, r);
        if (self.left) move = add(move, scale(r, -1));
        if (self.up) move.y += 1;
        if (self.down) move.y -= 1;

        const len = @sqrt(move.x * move.x + move.y * move.y + move.z * move.z);
        if (len > 0.0001) {
            const speed = self.move_speed * (if (self.sprint) self.sprint_mult else 1.0);
            const step = (speed * dt) / len;
            self.position.x += move.x * step;
            self.position.y += move.y * step;
            self.position.z += move.z * step;
        }
    }

    /// Write the controller's pose into a Camera3D (position + look target).
    pub fn applyTo(self: FirstPersonController, camera: *Camera3D) void {
        const f = self.forward();
        camera.position = self.position;
        camera.target = .{
            .x = self.position.x + f.x,
            .y = self.position.y + f.y,
            .z = self.position.z + f.z,
        };
    }

    fn add(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z };
    }
    fn scale(a: Vec3, s: f32) Vec3 {
        return .{ .x = a.x * s, .y = a.y * s, .z = a.z * s };
    }
};
