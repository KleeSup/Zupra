//! src/render/cubemap.zig
//!
//! Cubemap render target — the infrastructure for baking IBL maps. A single
//! cube image (6 faces, optionally mipped) that can be rendered INTO per face
//! and per mip (irradiance writes 6 faces; prefilter writes 6 faces x N mips,
//! one mip per roughness level) and SAMPLED afterward.
//!
//! Baking renders a fullscreen pass per face, reconstructing the world
//! direction for each pixel from the inverse of that face's view-projection
//! (same ray trick as the skybox). faceViewProjections() supplies those.

const std = @import("std");
const sg = @import("sokol").gfx;
const math = @import("../math.zig");
const zm = math.zm;

const Matrix = math.Matrix;
const Vec3 = math.Vec3;

const max_mips = 8;

pub const Cubemap = struct {
    img: sg.Image,
    sample_view: sg.View, // cube texture view, for sampling
    face_attachments: [6][max_mips]sg.View, // [face][mip], only [0..mip_count) valid
    size: u32,
    mip_count: u32,
    format: sg.PixelFormat,

    /// Allocate a renderable cubemap. `mip_count` > 1 for the prefilter map
    /// (each mip = a roughness level); 1 for irradiance / a plain env cube.
    pub fn initRenderTarget(size: u32, format: sg.PixelFormat, mip_count: u32) Cubemap {
        std.debug.assert(mip_count >= 1 and mip_count <= max_mips);

        const img = sg.makeImage(.{
            .type = .CUBE,
            .width = @intCast(size),
            .height = @intCast(size),
            .num_mipmaps = @intCast(mip_count),
            .pixel_format = format,
            .sample_count = 1,
            .usage = .{ .color_attachment = true },
        });

        var cm = Cubemap{
            .img = img,
            .sample_view = sg.makeView(.{ .texture = .{ .image = img } }),
            .face_attachments = undefined,
            .size = size,
            .mip_count = mip_count,
            .format = format,
        };

        var face: u32 = 0;
        while (face < 6) : (face += 1) {
            var mip: u32 = 0;
            while (mip < mip_count) : (mip += 1) {
                cm.face_attachments[face][mip] = sg.makeView(.{
                    .color_attachment = .{
                        .image = img,
                        .mip_level = @intCast(mip),
                        .slice = @intCast(face), // cube face index 0..5
                    },
                });
            }
        }
        return cm;
    }

    pub fn deinit(self: *Cubemap) void {
        var face: u32 = 0;
        while (face < 6) : (face += 1) {
            var mip: u32 = 0;
            while (mip < self.mip_count) : (mip += 1) {
                sg.destroyView(self.face_attachments[face][mip]);
            }
        }
        sg.destroyView(self.sample_view);
        sg.destroyImage(self.img);
    }

    /// Color-attachment view for rendering into one face at one mip level.
    pub fn faceAttachment(self: Cubemap, face: u32, mip: u32) sg.View {
        return self.face_attachments[face][mip];
    }

    /// Pixel size of a mip level (size >> mip, min 1).
    pub fn mipSize(self: Cubemap, mip: u32) u32 {
        const s = self.size >> @intCast(mip);
        return if (s == 0) 1 else s;
    }
};

/// View-projection per cube face (90° FOV, aspect 1). A bake pass renders a
/// fullscreen triangle into each face and reconstructs the sampling direction
/// from the inverse of that face's matrix.
///
/// Up-vectors follow the D3D LEFT-HANDED cube convention, matching LH
/// matrices (lookAtLh / perspectiveFovLh) and the D3D11 cube sampling order.
pub fn faceViewProjections() [6]Matrix {
    const proj = zm.perspectiveFovLh(0.5 * std.math.pi, 1.0, 0.1, 10.0); // 90°
    const eye = zm.f32x4(0, 0, 0, 1);

    const faces = [6][2]Vec3{
        .{ .{ .x = 1, .y = 0, .z = 0 }, .{ .x = 0, .y = 1, .z = 0 } }, // +X
        .{ .{ .x = -1, .y = 0, .z = 0 }, .{ .x = 0, .y = 1, .z = 0 } }, // -X
        .{ .{ .x = 0, .y = 1, .z = 0 }, .{ .x = 0, .y = 0, .z = -1 } }, // +Y
        .{ .{ .x = 0, .y = -1, .z = 0 }, .{ .x = 0, .y = 0, .z = 1 } }, // -Y
        .{ .{ .x = 0, .y = 0, .z = 1 }, .{ .x = 0, .y = 1, .z = 0 } }, // +Z
        .{ .{ .x = 0, .y = 0, .z = -1 }, .{ .x = 0, .y = 1, .z = 0 } }, // -Z
    };

    var result: [6]Matrix = undefined;
    for (faces, 0..) |f, i| {
        const dir = f[0];
        const up = f[1];
        // eye at origin, focus along dir (focus point = origin + dir = dir).
        const view = zm.lookAtLh(eye, zm.f32x4(dir.x, dir.y, dir.z, 1), zm.f32x4(up.x, up.y, up.z, 0));
        result[i] = zm.mul(view, proj);
    }
    return result;
}
