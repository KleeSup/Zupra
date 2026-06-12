//! src/render/fullscreen.zig
//!
//! Shared fullscreen-triangle vertex buffer for screen-space passes (deferred
//! lighting, tonemap/present, and later SSAO/SSR). A single triangle covering
//! the screen in clip space, non-indexed. UVs are chosen against the backend's
//! render-target origin so sampling a render target isn't vertically flipped.

const sg = @import("sokol").gfx;
const gfx = @import("../graphics/graphics.zig");
const Vertex2D = gfx.Vertex2D;

pub const FullscreenTriangle = struct {
    vbuf: sg.Buffer,

    pub fn init() FullscreenTriangle {
        const top_left = sg.queryFeatures().origin_top_left;
        const clip = [3][2]f32{ .{ -1, -1 }, .{ 3, -1 }, .{ -1, 3 } };
        var verts: [3]Vertex2D = undefined;
        for (clip, 0..) |p, i| {
            const u = 0.5 + 0.5 * p[0];
            const v = if (top_left) 0.5 - 0.5 * p[1] else 0.5 + 0.5 * p[1];
            verts[i] = .{ .pos = p, .uv = .{ u, v }, .color = 0xFFFFFFFF };
        }
        return .{ .vbuf = sg.makeBuffer(.{
            .usage = .{ .vertex_buffer = true, .immutable = true },
            .data = sg.asRange(&verts),
        }) };
    }

    pub fn deinit(self: FullscreenTriangle) void {
        sg.destroyBuffer(self.vbuf);
    }
};
