//! src/render/culling.zig
//!
//! Bounding volumes and frustum tests.
//!
//! Used first by the shadow system, which renders the same geometry into every
//! cascade and every cube face and would otherwise draw the whole scene once per
//! view. The same primitives are what general camera-frustum culling needs, so
//! they live here rather than inside the shadow renderer.
//!
//! Spheres rather than boxes: a sphere survives an arbitrary rotation without
//! needing to be refitted, the test against a plane is one dot product, and for
//! culling a slightly loose volume only costs the occasional draw that could
//! have been skipped. A false ACCEPT is free; only a false reject is a bug.

const std = @import("std");
const math = @import("../math.zig");

const Matrix = math.Matrix;
const Vec3 = math.Vec3;

pub const Sphere = struct {
    center: Vec3 = .{ .x = 0, .y = 0, .z = 0 },
    /// Negative means "empty" -- an empty sphere merges into anything without
    /// affecting it, and fails every intersection test.
    radius: f32 = -1.0,

    pub const empty: Sphere = .{};

    pub fn isEmpty(self: Sphere) bool {
        return self.radius < 0.0;
    }

    /// Fit around any slice of vertices carrying a `pos: [3]f32` field.
    ///
    /// Generic over the vertex type so this can read the caller's array in
    /// place: mesh vertices are handed straight to an immutable GPU buffer, and
    /// copying out just the positions to fit a sphere would mean an allocation
    /// on a path that has no reason to fail.
    ///
    /// Uses the AABB centre, not the centroid. A centroid drifts toward wherever
    /// vertices happen to be dense -- on a UV sphere, toward the poles -- and
    /// gives a measurably looser fit for no benefit.
    pub fn fromVertices(comptime T: type, verts: []const T) Sphere {
        if (verts.len == 0) return .empty;

        var min = verts[0].pos;
        var max = verts[0].pos;
        for (verts[1..]) |v| {
            for (0..3) |i| {
                min[i] = @min(min[i], v.pos[i]);
                max[i] = @max(max[i], v.pos[i]);
            }
        }

        const c = Vec3{
            .x = (min[0] + max[0]) * 0.5,
            .y = (min[1] + max[1]) * 0.5,
            .z = (min[2] + max[2]) * 0.5,
        };

        var r2: f32 = 0;
        for (verts) |v| {
            const dx = v.pos[0] - c.x;
            const dy = v.pos[1] - c.y;
            const dz = v.pos[2] - c.z;
            r2 = @max(r2, dx * dx + dy * dy + dz * dz);
        }
        return .{ .center = c, .radius = @sqrt(r2) };
    }

    /// Smallest sphere containing both. Not the minimal bounding sphere, but the
    /// standard conservative merge, which is all a cull test needs.
    pub fn merge(a: Sphere, b: Sphere) Sphere {
        if (a.isEmpty()) return b;
        if (b.isEmpty()) return a;

        const d = b.center.sub(a.center);
        const dist = d.length();

        // One already contains the other.
        if (dist + b.radius <= a.radius) return a;
        if (dist + a.radius <= b.radius) return b;

        const r = (dist + a.radius + b.radius) * 0.5;
        // Walk from a's centre toward b's by however much the new radius grew.
        const t = if (dist > 1e-6) (r - a.radius) / dist else 0.0;
        return .{ .center = a.center.add(d.mul(t)), .radius = r };
    }

    /// Object space -> world space under a model matrix.
    ///
    /// The radius scales by the LARGEST of the three axis scales, because a
    /// sphere has no axes to scale independently: under a non-uniform scale the
    /// true shape is an ellipsoid, and the enclosing sphere is set by its
    /// longest semi-axis. Taking the largest keeps the volume conservative.
    pub fn transform(self: Sphere, m: Matrix) Sphere {
        if (self.isEmpty()) return .empty;

        const c = transformPoint(m, self.center);

        // Row-vector convention: rows 0..2 are the transformed basis vectors, so
        // their lengths are the axis scales.
        const sx = @sqrt(m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[0][2] * m[0][2]);
        const sy = @sqrt(m[1][0] * m[1][0] + m[1][1] * m[1][1] + m[1][2] * m[1][2]);
        const sz = @sqrt(m[2][0] * m[2][0] + m[2][1] * m[2][1] + m[2][2] * m[2][2]);

        return .{ .center = c, .radius = self.radius * @max(sx, @max(sy, sz)) };
    }
};

/// Six inward-facing planes, each stored as (a, b, c, d) with the convention
/// that a point is inside when a*x + b*y + c*z + d >= 0.
pub const Frustum = struct {
    planes: [6][4]f32 = .{.{ 0, 0, 0, 0 }} ** 6,

    /// Extract the planes directly from a view-projection matrix (the
    /// Gribb-Hartmann method). Works for ANY projection the matrix encodes --
    /// a cascade's orthographic box, a spot's perspective cone and a cube face's
    /// 90-degree frustum all come out of the same code, which is the reason to
    /// cull this way rather than special-casing each light type.
    ///
    /// Two conventions are baked in and both matter. The engine is row-vector
    /// (clip = v * M), so a clip component is a dot product with a COLUMN of M,
    /// and each column is therefore already a plane equation in x, y, z, 1.
    /// And clip depth runs 0..w, not -w..w, so the near plane is column 2 alone
    /// rather than column 2 plus column 3.
    pub fn fromViewProj(m: Matrix) Frustum {
        // Each column of M gives the coefficients of one clip component as a
        // function of (x, y, z, 1). Written out rather than indexed in a loop
        // because Matrix rows are @Vector(4, f32) and a vector index has to be
        // comptime known.
        const cx = [4]f32{ m[0][0], m[1][0], m[2][0], m[3][0] };
        const cy = [4]f32{ m[0][1], m[1][1], m[2][1], m[3][1] };
        const cz = [4]f32{ m[0][2], m[1][2], m[2][2], m[3][2] };
        const cw = [4]f32{ m[0][3], m[1][3], m[2][3], m[3][3] };

        var f = Frustum{};
        f.planes[0] = addv(cw, cx); //  clip.x >= -clip.w   left
        f.planes[1] = subv(cw, cx); //  clip.x <=  clip.w   right
        f.planes[2] = addv(cw, cy); //  clip.y >= -clip.w   bottom
        f.planes[3] = subv(cw, cy); //  clip.y <=  clip.w   top
        f.planes[4] = cz; //            clip.z >=  0        near
        f.planes[5] = subv(cw, cz); //  clip.z <=  clip.w   far

        // Normalise so plane-to-point distances are real world distances, which
        // is what makes the radius comparison below meaningful.
        for (&f.planes) |*p| {
            const len = @sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
            if (len > 1e-6) {
                const inv = 1.0 / len;
                p[0] *= inv;
                p[1] *= inv;
                p[2] *= inv;
                p[3] *= inv;
            }
        }
        return f;
    }

    /// Conservative: returns true if the sphere might be visible. A sphere is
    /// rejected only when it lies fully behind one plane -- which can still let
    /// through a sphere near a corner that touches no plane's interior, and
    /// that is the correct trade. An extra draw costs a little time; a wrongly
    /// rejected caster loses its shadow.
    pub fn intersectsSphere(self: Frustum, s: Sphere) bool {
        if (s.isEmpty()) return false;
        for (self.planes) |p| {
            const dist = p[0] * s.center.x + p[1] * s.center.y + p[2] * s.center.z + p[3];
            if (dist < -s.radius) return false;
        }
        return true;
    }
};

/// Row-vector point transform: (x, y, z, 1) * M, dropping w. Local rather than
/// pulled from math.zig so this module has no dependency beyond the Matrix type.
fn transformPoint(m: Matrix, p: Vec3) Vec3 {
    return .{
        .x = p.x * m[0][0] + p.y * m[1][0] + p.z * m[2][0] + m[3][0],
        .y = p.x * m[0][1] + p.y * m[1][1] + p.z * m[2][1] + m[3][1],
        .z = p.x * m[0][2] + p.y * m[1][2] + p.z * m[2][2] + m[3][2],
    };
}

fn addv(a: [4]f32, b: [4]f32) [4]f32 {
    return .{ a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3] };
}

fn subv(a: [4]f32, b: [4]f32) [4]f32 {
    return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3] };
}
