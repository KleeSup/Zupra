//! src/render/camera3d.zig
//!
//! 3D perspective camera. Produces the view-projection matrix the mesh shader
//! consumes. Conventions match Camera2D so the whole engine stays coherent:
//!
//!   * Left-handed, [0,1] depth (zmath's perspectiveFovLh / lookAtLh, the
//!     non-Gl variants), represents the clip space sokol normalizes to on every
//!     backend, the same family as Camera2D's orthographicOffCenterLh.
//!   * Same matrix rule: zmath is row-major/row-vector, GLSL reads column-major,
//!     so upload viewProjection() directly (no transpose) and the shader's
//!     `mvp * vec4(pos,1)` reproduces the transform. Compose mul(view, proj).

const math = @import("../math.zig");
const zm = math.zm;
const Vec3 = math.Vec3;
const Matrix = math.Matrix;

pub const Camera3D = struct {
    position: Vec3 = .{ .x = 0, .y = 0, .z = -5 },
    target: Vec3 = .{ .x = 0, .y = 0, .z = 0 },
    up: Vec3 = .{ .x = 0, .y = 1, .z = 0 },
    fov_y: f32 = 60.0 * math.DEG_TO_RAD, // radians
    aspect: f32 = 16.0 / 9.0,
    near: f32 = 0.1,
    far: f32 = 1000.0,

    /// Sub-pixel projection offset in NDC, for temporal anti-aliasing.
    /// This value should be zero until TAA is enabled in a used SceneRenderer.
    ///
    /// Idea: It gets applied to the projection and must shift where the
    /// scene lands on the pixel grid without changing where the camera is (or
    /// reprojection would have to undo it again).
    jitter: [2]f32 = .{ 0, 0 },

    pub fn init(aspect: f32) Camera3D {
        return .{ .aspect = aspect };
    }

    /// Set aspect from the framebuffer size. Call each frame or when window sizes change with
    /// sokol.app.widthf()/heightf().
    pub fn setViewport(self: *Camera3D, w: f32, h: f32) void {
        self.aspect = if (h != 0) w / h else 1.0;
    }

    pub fn setFovDegrees(self: *Camera3D, degrees: f32) void {
        self.fov_y = degrees * math.DEG_TO_RAD;
    }

    pub fn lookAt(self: *Camera3D, point: Vec3) void {
        self.target = point;
    }

    /// World->view. Left-handed lookAt.
    pub fn view(self: Camera3D) Matrix {
        return zm.lookAtLh(v4(self.position, 1), v4(self.target, 1), v4(self.up, 0));
    }

    /// View->clip. [0,1] depth, matches sokol.
    pub fn projection(self: Camera3D) Matrix {
        var p = zm.perspectiveFovLh(self.fov_y, self.aspect, self.near, self.far);
        // w == view z here, so a z-proportional offset becomes a constant NDC shift after the divide.
        p[2][0] += self.jitter[0];
        p[2][1] += self.jitter[1];
        return p;
    }

    /// The projection without the jitter. The jitter is
    /// a rendering offset, not part of where a surface actually is, and feeding
    /// the jittered matrix to TAA makes the history chase the jitter sequence
    /// instead of the camera (which is why it needs Reprojection).
    pub fn unjitteredViewProjection(self: Camera3D) Matrix {
        var c = self;
        c.jitter = .{ 0, 0 };
        return zm.mul(c.view(), c.projection());
    }

    /// The matrix the mesh shader's view_proj uniform wants. Upload directly.
    pub fn viewProjection(self: Camera3D) Matrix {
        return zm.mul(self.view(), self.projection());
    }

    /// Normalized forward direction (from position toward target).
    pub fn forward(self: Camera3D) Vec3 {
        return self.target.sub(self.position).normalize();
    }
};

/// Promote a Vec3 to the F32x4 zmath expects (w: 1 for points, 0 for directions).
inline fn v4(p: Vec3, w: f32) zm.Vec {
    return zm.f32x4(p.x, p.y, p.z, w);
}
