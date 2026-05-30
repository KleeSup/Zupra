// --- Animation ---

pub const AnimationMode = enum {
    /// Play the animation once and stop at the last frame.
    NORMAL,
    /// Loop the animation, starting again at the first frame after reaching the last one.
    LOOP,
    /// Play the animation in reverse, starting at the last frame and stopping at the first one.
    REVERSE,
    /// Loop the animation in reverse, starting at the last frame and looping back to the last one after reaching the first one.
    REVERSE_LOOP,
};

/// Like Animation, but it already includes the state time, so you don't have to manage it yourself.
/// Just call update() with the delta time and it will return the current frame.
pub fn StateAnimation(comptime T: type) type {
    return struct {
        animation: Animation(T),
        state_time: f32 = 0,
        pub fn init(animation: Animation(T)) @This() {
            return .{ .animation = animation };
        }
        pub fn update(self: @This(), delta: f32) *T {
            self.state_time += delta;
            return self.getFrame();
        }
        pub fn getFrame(self: @This()) *T {
            return self.animation.getFrame(self.state_time);
        }
    };
}

/// A simple animation struct that holds an array of frames and a frame duration.
/// It also requires a mode that determines how the animation should play.
pub fn Animation(comptime T: type) type {
    return struct {
        frames: []T,
        mode: AnimationMode = AnimationMode.NORMAL,
        frameDuration: f32 = 1.0,
        invFrameDuration: f32 = 0,

        pub fn init(frames: []T, frameDuration: f32) @This() {
            return .{
                .mode = AnimationMode.NORMAL,
                .frames = frames,
                .frameDuration = frameDuration,
                .invFrameDuration = 1.0 / frameDuration,
            };
        }

        pub fn getFrame(self: @This(), stateTime: f32) *T {
            const frame_count: i32 = @intCast(self.frames.len);
            const raw_frame: i32 = @intFromFloat(stateTime * self.invFrameDuration);

            const index: i32 = switch (self.mode) {
                .NORMAL => blk: {
                    if (raw_frame >= frame_count)
                        break :blk frame_count - 1;
                    break :blk raw_frame;
                },

                .LOOP => @mod(raw_frame, frame_count),

                .REVERSE => blk: {
                    if (raw_frame >= frame_count)
                        break :blk 0;
                    break :blk (frame_count - 1) - raw_frame;
                },

                .REVERSE_LOOP => blk: {
                    const i = @mod(raw_frame, frame_count);
                    break :blk (frame_count - 1) - i;
                },
            };

            return &self.frames[@intCast(index)];
        }
    };
}
