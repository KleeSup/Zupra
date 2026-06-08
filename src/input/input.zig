const std = @import("std");
const math = @import("../math.zig");
const sokol = @import("sokol");
const EnumSet = std.EnumSet;

pub const Keycode = sokol.app.Keycode;
pub const Mousebutton = sokol.app.Mousebutton;

var focused = true;
var pressed_button = EnumSet(Mousebutton).empty;
var just_pressed_button = EnumSet(Mousebutton).empty;
var pressed_keys = EnumSet(Keycode).empty;
var just_pressed_keys = EnumSet(Keycode).empty;
var mouse_delta: math.Vec2 = .{ .x = 0, .y = 0 };
var mouse_scroll: math.Vec2 = .{ .x = 0, .y = 0 };
pub var mouseX: f32 = 0;
pub var mouseY: f32 = 0;

pub fn _updateFrame() void {
    just_pressed_keys = EnumSet(Keycode).empty;
    just_pressed_button = EnumSet(Mousebutton).empty;
    mouse_delta = .{ .x = 0, .y = 0 };
    mouse_scroll = .{ .x = 0, .y = 0 };
}

pub fn _updateEvent(event: *const sokol.app.Event) void {
    switch (event.type) {
        .FOCUSED => {
            focused = true;
        },
        .UNFOCUSED => {
            focused = false;
        },
        .KEY_DOWN => {
            pressed_keys.insert(event.key_code);
            if (!event.key_repeat) just_pressed_keys.insert(event.key_code);
        },
        .KEY_UP => {
            pressed_keys.remove(event.key_code);
            just_pressed_keys.remove(event.key_code);
        },
        .MOUSE_MOVE => {
            if (!focused) return;
            mouseX = event.mouse_x;
            mouseY = event.mouse_y;
            mouse_delta.x = event.mouse_dx;
            mouse_delta.y = event.mouse_dy;
        },
        .MOUSE_DOWN => {
            pressed_button.insert(event.mouse_button);
            if (!pressed_button.contains(event.mouse_button)) just_pressed_button.insert(event.mouse_button);
        },
        .MOUSE_UP => {
            just_pressed_button.remove(event.mouse_button);
            pressed_button.remove(event.mouse_button);
        },
        .MOUSE_SCROLL => {
            if (!focused) return;
            mouse_scroll.x = event.scroll_x;
            mouse_scroll.y = event.scroll_y;
        },
        else => {},
    }
}

pub fn isKeyPressed(key: Keycode) bool {
    return pressed_keys.contains(key);
}

pub fn isKeyJustPressed(key: Keycode) bool {
    return just_pressed_keys.contains(key);
}

pub fn isButtonPressed(button: Mousebutton) bool {
    return pressed_button.contains(button);
}
pub fn isButtonJustPressed(button: Mousebutton) bool {
    return just_pressed_button.contains(button);
}

pub fn getMouseDelta() math.Vec2 {
    return mouse_delta;
}

pub fn getMouseScroll() math.Vec2 {
    return mouse_scroll;
}
