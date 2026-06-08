//! 2D camera. Produces the `mvp` matrix the default sprite shader consumes
//! (vs_params.mvp). This is where the SpriteBatch coordinate convention lives:
//!
//!   * Pixel space: world units are screen pixels.
//!   * Origin top-left, y increases downward (standard sprite convention).
//!   * With the default target/offset/zoom, world (x,y) maps straight to
//!     pixel (x,y) — i.e. no camera transform until you set one.
//!
//! MATRIX LAYOUT NOTE: zmath stores matrices row-major (row-vector math,
//! p' = p * M). GLSL uniforms are read column-major, so an uploaded zmath
//! matrix is seen transposed by the shader — which is exactly what makes
//! `gl_Position = mvp * vec4(pos,0,1)` reproduce the row-vector transform.
//! So: upload the result of viewProjection() directly, do NOT transpose, and
//! compose as mul(view, proj) (view applied first).

const math = @import("../math.zig");
const zm = math.zm;
const Vec2 = math.Vec2;
const Matrix = math.Matrix;

pub const Camera2D = struct {
    /// World point that maps onto `offset`.
    target: Vec2 = .{ .x = 0, .y = 0 },
    /// Screen-pixel point that `target` maps to. Set to half the viewport to
    /// center the target on screen. Leave it at (0,0) for world==screen pixels.
    offset: Vec2 = .{ .x = 0, .y = 0 },
    rotation: f32 = 0, // radians
    zoom: f32 = 1.0,
    viewport_width: f32,
    viewport_height: f32,
    near: f32 = -1.0,
    far: f32 = 1.0,

    pub fn init(viewport_width: f32, viewport_height: f32) Camera2D {
        return .{ .viewport_width = viewport_width, .viewport_height = viewport_height };
    }

    /// Call on window resize.
    pub fn setViewport(self: *Camera2D, w: f32, h: f32) void {
        self.viewport_width = w;
        self.viewport_height = h;
    }

    /// Moves the camera by the given `move` vector.
    pub fn translate(self: *Camera2D, move: Vec2) void {
        self.target.x += move.x;
        self.target.y += move.y;
    }

    /// Pixel space -> clip space. top=0, bottom=height gives the y-down flip.
    pub fn projection(self: Camera2D) Matrix {
        return zm.orthographicOffCenterLh(
            0,
            self.viewport_width,
            0, // top
            self.viewport_height, // bottom
            self.near,
            self.far,
        );
    }

    /// World space -> pixel space: subtract target, scale by zoom, rotate,
    /// add offset. (Row-vector order: leftmost is applied first.)
    pub fn view(self: Camera2D) Matrix {
        const to_origin = zm.translation(-self.target.x, -self.target.y, 0);
        const scale = zm.scaling(self.zoom, self.zoom, 1);
        const rot = zm.rotationZ(self.rotation);
        const to_offset = zm.translation(self.offset.x, self.offset.y, 0);
        return zm.mul(zm.mul(zm.mul(to_origin, scale), rot), to_offset);
    }

    /// The matrix to upload into vs_params.mvp.
    pub fn viewProjection(self: Camera2D) Matrix {
        return zm.mul(self.view(), self.projection());
    }

    /// World point -> screen pixel (no matrix roundtrip; handy for UI/anchoring).
    pub fn worldToScreen(self: Camera2D, p: Vec2) Vec2 {
        const d = Vec2{
            .x = (p.x - self.target.x) * self.zoom,
            .y = (p.y - self.target.y) * self.zoom,
        };
        const r = rotateVec2(d, self.rotation);
        return .{ .x = r.x + self.offset.x, .y = r.y + self.offset.y };
    }

    /// Screen pixel -> world point (e.g. mouse picking). Inverse of the above.
    pub fn screenToWorld(self: Camera2D, s: Vec2) Vec2 {
        const d = Vec2{ .x = s.x - self.offset.x, .y = s.y - self.offset.y };
        const r = rotateVec2(d, -self.rotation);
        return .{
            .x = r.x / self.zoom + self.target.x,
            .y = r.y / self.zoom + self.target.y,
        };
    }
};

fn rotateVec2(v: Vec2, angle: f32) Vec2 {
    const c = @cos(angle);
    const s = @sin(angle);
    return .{ .x = v.x * c - v.y * s, .y = v.x * s + v.y * c };
}
