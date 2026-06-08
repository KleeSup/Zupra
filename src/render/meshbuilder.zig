//! src/render/meshbuilder.zig
//!
//! Procedural primitive meshes (cube, sphere, plane) producing a GPU-ready
//! Mesh. Ported from the old ModelBuilder, adapted to the new Vertex3D
//! (pos/normal/uv — no color/tangent yet; tangent arrives with normal mapping).
//!
//! Index width: cube/plane always fit u16, so they use it. Sphere auto-selects
//! u16 when the vertex count fits (<= 65536) else u32 — exercising the per-mesh
//! index-width choice. Mesh.init copies into immutable GPU buffers, so all the
//! CPU scratch here is freed before returning.
//!
//! Winding is consistent CCW.

const std = @import("std");
const gfx = @import("../graphics/graphics.zig");
const mesh_mod = @import("mesh.zig");

const Vertex3D = gfx.Vertex3D;
const Mesh = mesh_mod.Mesh;

pub const MeshBuilder = struct {
    /// Axis-aligned box centered at origin. 24 verts (per-face normals), u16.
    pub fn cube(w: f32, h: f32, d: f32) Mesh {
        const x = w * 0.5;
        const y = h * 0.5;
        const z = d * 0.5;

        const verts = [24]Vertex3D{
            // +X
            .{ .pos = .{ x, -y, -z }, .normal = .{ 1, 0, 0 }, .uv = .{ 0, 0 } },
            .{ .pos = .{ x, y, -z }, .normal = .{ 1, 0, 0 }, .uv = .{ 0, 1 } },
            .{ .pos = .{ x, y, z }, .normal = .{ 1, 0, 0 }, .uv = .{ 1, 1 } },
            .{ .pos = .{ x, -y, z }, .normal = .{ 1, 0, 0 }, .uv = .{ 1, 0 } },
            // -X
            .{ .pos = .{ -x, -y, z }, .normal = .{ -1, 0, 0 }, .uv = .{ 0, 0 } },
            .{ .pos = .{ -x, y, z }, .normal = .{ -1, 0, 0 }, .uv = .{ 0, 1 } },
            .{ .pos = .{ -x, y, -z }, .normal = .{ -1, 0, 0 }, .uv = .{ 1, 1 } },
            .{ .pos = .{ -x, -y, -z }, .normal = .{ -1, 0, 0 }, .uv = .{ 1, 0 } },
            // +Y
            .{ .pos = .{ -x, y, -z }, .normal = .{ 0, 1, 0 }, .uv = .{ 0, 0 } },
            .{ .pos = .{ -x, y, z }, .normal = .{ 0, 1, 0 }, .uv = .{ 0, 1 } },
            .{ .pos = .{ x, y, z }, .normal = .{ 0, 1, 0 }, .uv = .{ 1, 1 } },
            .{ .pos = .{ x, y, -z }, .normal = .{ 0, 1, 0 }, .uv = .{ 1, 0 } },
            // -Y
            .{ .pos = .{ -x, -y, z }, .normal = .{ 0, -1, 0 }, .uv = .{ 0, 0 } },
            .{ .pos = .{ -x, -y, -z }, .normal = .{ 0, -1, 0 }, .uv = .{ 0, 1 } },
            .{ .pos = .{ x, -y, -z }, .normal = .{ 0, -1, 0 }, .uv = .{ 1, 1 } },
            .{ .pos = .{ x, -y, z }, .normal = .{ 0, -1, 0 }, .uv = .{ 1, 0 } },
            // +Z
            .{ .pos = .{ x, -y, z }, .normal = .{ 0, 0, 1 }, .uv = .{ 0, 0 } },
            .{ .pos = .{ x, y, z }, .normal = .{ 0, 0, 1 }, .uv = .{ 0, 1 } },
            .{ .pos = .{ -x, y, z }, .normal = .{ 0, 0, 1 }, .uv = .{ 1, 1 } },
            .{ .pos = .{ -x, -y, z }, .normal = .{ 0, 0, 1 }, .uv = .{ 1, 0 } },
            // -Z
            .{ .pos = .{ -x, -y, -z }, .normal = .{ 0, 0, -1 }, .uv = .{ 0, 0 } },
            .{ .pos = .{ -x, y, -z }, .normal = .{ 0, 0, -1 }, .uv = .{ 0, 1 } },
            .{ .pos = .{ x, y, -z }, .normal = .{ 0, 0, -1 }, .uv = .{ 1, 1 } },
            .{ .pos = .{ x, -y, -z }, .normal = .{ 0, 0, -1 }, .uv = .{ 1, 0 } },
        };

        var indices: [36]u16 = undefined;
        var f: u16 = 0;
        while (f < 6) : (f += 1) {
            const b = f * 4;
            const o = f * 6;
            indices[o + 0] = b + 0;
            indices[o + 1] = b + 1;
            indices[o + 2] = b + 2;
            indices[o + 3] = b + 0;
            indices[o + 4] = b + 2;
            indices[o + 5] = b + 3;
        }

        return Mesh.init(&verts, .{ .u16 = &indices });
    }

    /// Flat XZ plane centered at origin, facing +Y. Single-sided (visible both
    /// ways under cull = .NONE). 4 verts, u16.
    pub fn plane(w: f32, d: f32) Mesh {
        const x = w * 0.5;
        const z = d * 0.5;
        const verts = [4]Vertex3D{
            .{ .pos = .{ -x, 0, z }, .normal = .{ 0, 1, 0 }, .uv = .{ 0, 0 } },
            .{ .pos = .{ x, 0, z }, .normal = .{ 0, 1, 0 }, .uv = .{ 1, 0 } },
            .{ .pos = .{ x, 0, -z }, .normal = .{ 0, 1, 0 }, .uv = .{ 1, 1 } },
            .{ .pos = .{ -x, 0, -z }, .normal = .{ 0, 1, 0 }, .uv = .{ 0, 1 } },
        };
        const indices = [6]u16{ 0, 1, 2, 0, 2, 3 };
        return Mesh.init(&verts, .{ .u16 = &indices });
    }

    /// UV sphere with smooth (radial) normals. `rings` = latitude divisions,
    /// `slices` = longitude divisions.
    pub fn sphere(alloc: std.mem.Allocator, radius: f32, rings: u32, slices: u32) !Mesh {
        const num_verts = (rings + 1) * (slices + 1);
        const num_inds = rings * slices * 6;

        const verts = try alloc.alloc(Vertex3D, num_verts);
        defer alloc.free(verts);
        const idx = try alloc.alloc(u32, num_inds);
        defer alloc.free(idx);

        var v: usize = 0;
        var lat: u32 = 0;
        while (lat <= rings) : (lat += 1) {
            const fv = @as(f32, @floatFromInt(lat)) / @as(f32, @floatFromInt(rings));
            const theta = fv * std.math.pi;
            const sin_t = @sin(theta);
            const cos_t = @cos(theta);

            var lon: u32 = 0;
            while (lon <= slices) : (lon += 1) {
                const fu = @as(f32, @floatFromInt(lon)) / @as(f32, @floatFromInt(slices));
                const phi = fu * 2.0 * std.math.pi;
                const nx = @cos(phi) * sin_t;
                const ny = cos_t;
                const nz = @sin(phi) * sin_t;
                verts[v] = .{
                    .pos = .{ nx * radius, ny * radius, nz * radius },
                    .normal = .{ nx, ny, nz },
                    .uv = .{ fu, fv },
                };
                v += 1;
            }
        }

        var i: usize = 0;
        lat = 0;
        while (lat < rings) : (lat += 1) {
            var lon: u32 = 0;
            while (lon < slices) : (lon += 1) {
                const first = lat * (slices + 1) + lon;
                const second = first + (slices + 1);
                idx[i + 0] = first;
                idx[i + 1] = first + 1;
                idx[i + 2] = second;
                idx[i + 3] = first + 1;
                idx[i + 4] = second + 1;
                idx[i + 5] = second;
                i += 6;
            }
        }

        // Pick the narrowest index width that fits this vertex count.
        if (num_verts <= 65536) {
            const idx16 = try alloc.alloc(u16, num_inds);
            defer alloc.free(idx16);
            for (idx, 0..) |val, k| idx16[k] = @intCast(val);
            return Mesh.init(verts, .{ .u16 = idx16 });
        } else {
            return Mesh.init(verts, .{ .u32 = idx });
        }
    }
};
