//! src/render/meshbuilder.zig
//!
//! Procedural primitive meshes (cube, sphere, plane) producing a GPU-ready
//! Mesh.
//!
//! Index width: cube/plane use u16. Sphere auto-selects
//! u16 when the vertex count fits (<= 65536) else u32.
//! Mesh.init copies into immutable GPU buffers, so all the CPU scratch here is freed before returning.

const std = @import("std");
const gfx = @import("../graphics/graphics.zig");
const mesh_mod = @import("mesh.zig");

const Vertex3D = gfx.Vertex3D;
const Mesh = mesh_mod.Mesh;

/// Append two front-facing (CCW) triangles for a quad to `out`, advancing `n`.
/// v0..v3 are the quad's four corner indices listed around the face. The single
/// source of truth for mesh winding.
fn emitQuad(comptime T: type, out: []T, n: *usize, v0: T, v1: T, v2: T, v3: T) void {
    out[n.* + 0] = v0;
    out[n.* + 1] = v2;
    out[n.* + 2] = v1;
    out[n.* + 3] = v0;
    out[n.* + 4] = v3;
    out[n.* + 5] = v2;
    n.* += 6;
}

pub const MeshBuilder = struct {
    /// Axis-aligned box centered at origin. 24 verts (per-face normals +
    /// tangents), u16. Tangent per face = the world direction of +U.
    pub fn cube(w: f32, h: f32, d: f32) Mesh {
        const x = w * 0.5;
        const y = h * 0.5;
        const z = d * 0.5;

        // Per-face tangent (direction of increasing U), w = +1 handedness.
        const tpx = [4]f32{ 0, 0, 1, 1 }; // +X face: +U -> +Z
        const tnx = [4]f32{ 0, 0, -1, 1 }; // -X face: +U -> -Z
        const tpy = [4]f32{ 1, 0, 0, 1 }; // +Y face: +U -> +X
        const tny = [4]f32{ 1, 0, 0, 1 }; // -Y face: +U -> +X
        const tpz = [4]f32{ -1, 0, 0, 1 }; // +Z face: +U -> -X
        const tnz = [4]f32{ 1, 0, 0, 1 }; // -Z face: +U -> +X

        const verts = [24]Vertex3D{
            // +X
            .{ .pos = .{ x, -y, -z }, .normal = .{ 1, 0, 0 }, .uv = .{ 0, 0 }, .tangent = tpx },
            .{ .pos = .{ x, y, -z }, .normal = .{ 1, 0, 0 }, .uv = .{ 0, 1 }, .tangent = tpx },
            .{ .pos = .{ x, y, z }, .normal = .{ 1, 0, 0 }, .uv = .{ 1, 1 }, .tangent = tpx },
            .{ .pos = .{ x, -y, z }, .normal = .{ 1, 0, 0 }, .uv = .{ 1, 0 }, .tangent = tpx },
            // -X
            .{ .pos = .{ -x, -y, z }, .normal = .{ -1, 0, 0 }, .uv = .{ 0, 0 }, .tangent = tnx },
            .{ .pos = .{ -x, y, z }, .normal = .{ -1, 0, 0 }, .uv = .{ 0, 1 }, .tangent = tnx },
            .{ .pos = .{ -x, y, -z }, .normal = .{ -1, 0, 0 }, .uv = .{ 1, 1 }, .tangent = tnx },
            .{ .pos = .{ -x, -y, -z }, .normal = .{ -1, 0, 0 }, .uv = .{ 1, 0 }, .tangent = tnx },
            // +Y
            .{ .pos = .{ -x, y, -z }, .normal = .{ 0, 1, 0 }, .uv = .{ 0, 0 }, .tangent = tpy },
            .{ .pos = .{ -x, y, z }, .normal = .{ 0, 1, 0 }, .uv = .{ 0, 1 }, .tangent = tpy },
            .{ .pos = .{ x, y, z }, .normal = .{ 0, 1, 0 }, .uv = .{ 1, 1 }, .tangent = tpy },
            .{ .pos = .{ x, y, -z }, .normal = .{ 0, 1, 0 }, .uv = .{ 1, 0 }, .tangent = tpy },
            // -Y
            .{ .pos = .{ -x, -y, z }, .normal = .{ 0, -1, 0 }, .uv = .{ 0, 0 }, .tangent = tny },
            .{ .pos = .{ -x, -y, -z }, .normal = .{ 0, -1, 0 }, .uv = .{ 0, 1 }, .tangent = tny },
            .{ .pos = .{ x, -y, -z }, .normal = .{ 0, -1, 0 }, .uv = .{ 1, 1 }, .tangent = tny },
            .{ .pos = .{ x, -y, z }, .normal = .{ 0, -1, 0 }, .uv = .{ 1, 0 }, .tangent = tny },
            // +Z
            .{ .pos = .{ x, -y, z }, .normal = .{ 0, 0, 1 }, .uv = .{ 0, 0 }, .tangent = tpz },
            .{ .pos = .{ x, y, z }, .normal = .{ 0, 0, 1 }, .uv = .{ 0, 1 }, .tangent = tpz },
            .{ .pos = .{ -x, y, z }, .normal = .{ 0, 0, 1 }, .uv = .{ 1, 1 }, .tangent = tpz },
            .{ .pos = .{ -x, -y, z }, .normal = .{ 0, 0, 1 }, .uv = .{ 1, 0 }, .tangent = tpz },
            // -Z
            .{ .pos = .{ -x, -y, -z }, .normal = .{ 0, 0, -1 }, .uv = .{ 0, 0 }, .tangent = tnz },
            .{ .pos = .{ -x, y, -z }, .normal = .{ 0, 0, -1 }, .uv = .{ 0, 1 }, .tangent = tnz },
            .{ .pos = .{ x, y, -z }, .normal = .{ 0, 0, -1 }, .uv = .{ 1, 1 }, .tangent = tnz },
            .{ .pos = .{ x, -y, -z }, .normal = .{ 0, 0, -1 }, .uv = .{ 1, 0 }, .tangent = tnz },
        };

        var indices: [36]u16 = undefined;
        var n: usize = 0;
        var f: u16 = 0;
        while (f < 6) : (f += 1) {
            const b: u16 = f * 4;
            emitQuad(u16, &indices, &n, b + 0, b + 1, b + 2, b + 3);
        }

        return Mesh.init(&verts, .{ .u16 = &indices });
    }

    /// Flat XZ plane centered at origin, facing +Y. Tangent = +X (direction of +U).
    pub fn plane(w: f32, d: f32) Mesh {
        const x = w * 0.5;
        const z = d * 0.5;
        const tan = [4]f32{ 1, 0, 0, 1 };
        const verts = [4]Vertex3D{
            .{ .pos = .{ -x, 0, z }, .normal = .{ 0, 1, 0 }, .uv = .{ 0, 0 }, .tangent = tan },
            .{ .pos = .{ x, 0, z }, .normal = .{ 0, 1, 0 }, .uv = .{ 1, 0 }, .tangent = tan },
            .{ .pos = .{ x, 0, -z }, .normal = .{ 0, 1, 0 }, .uv = .{ 1, 1 }, .tangent = tan },
            .{ .pos = .{ -x, 0, -z }, .normal = .{ 0, 1, 0 }, .uv = .{ 0, 1 }, .tangent = tan },
        };
        var indices: [6]u16 = undefined;
        var n: usize = 0;
        emitQuad(u16, &indices, &n, 0, 1, 2, 3);
        return Mesh.init(&verts, .{ .u16 = &indices });
    }

    /// UV sphere with smooth normals. Tangent = d(pos)/d(phi) = the +longitude
    /// direction, which is exactly (-sin phi, 0, cos phi) (matches uv.u = phi).
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
                const sin_p = @sin(phi);
                const cos_p = @cos(phi);
                const nx = cos_p * sin_t;
                const ny = cos_t;
                const nz = sin_p * sin_t;
                verts[v] = .{
                    .pos = .{ nx * radius, ny * radius, nz * radius },
                    .normal = .{ nx, ny, nz },
                    .uv = .{ fu, fv },
                    .tangent = .{ -sin_p, 0, cos_p, 1 }, // d pos / d phi, normalized
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
                emitQuad(u32, idx, &i, second, first, first + 1, second + 1);
            }
        }

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
