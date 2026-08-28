const std = @import("std");
const shader = @import("shader.zig");

pub const pipeline = @import("pipeline.zig");
pub const texture = @import("texture.zig");
pub const ShaderProgram = shader.ShaderProgram;
pub const ShaderOptions = shader.Options;
pub const ShaderSlots = shader.Slots;

// --- Vertex types with canonical layouts ---
// Attribute SLOT order is a contract: slot 0 = first field, slot 1 = second,
// etc. Default shaders (and any custom shader that wants to plug into a given
// layout) must declare their vertex inputs in this same order so the shdc
// ATTR_* slot indices line up.

/// 2D batch / sprite vertex. Color is packed RGBA8 (UBYTE4N) to keep the vertex small.
pub const Vertex2D = extern struct {
    pos: [2]f32,
    uv: [2]f32,
    color: u32,
};

/// Standard 3D mesh vertex.
pub const Vertex3D = extern struct {
    pos: [3]f32,
    normal: [3]f32,
    uv: [2]f32,
    tangent: [4]f32 = .{ 1, 0, 0, 1 },
    uv1: [2]f32 = .{ 0, 0 }, // glTF TEXCOORD_1 (baked AO, lightmaps, detail)
};

/// glTF skinned mesh vertex. Kept separate from `Vertex3D` so static geometry
/// does not pay for joint data in GPU bandwidth or storage.
///
/// `joints` are indices into the skin palette attached to the draw, not global
/// node IDs. Four 16-bit indices cover the glTF core JOINTS_0 accessor range;
/// weights are decoded to floats by the loader and normalized before upload.
pub const VertexSkinned3D = extern struct {
    pos: [3]f32,
    normal: [3]f32,
    uv: [2]f32,
    tangent: [4]f32 = .{ 1, 0, 0, 1 },
    uv1: [2]f32 = .{ 0, 0 },
    joints: [4]u16 = .{ 0, 0, 0, 0 },
    weights: [4]f32 = .{ 1, 0, 0, 0 },
};

/// Debug-draw vertex (lines / wire shapes), 2D and 3D.
pub const VertexDebug = extern struct {
    pos: [3]f32,
    color: u32,
};

pub const IndexType = enum {
    u16,
    u32,
    pub fn size(self: IndexType) u32 {
        return switch (self) {
            .u16 => 2,
            .u32 => 4,
        };
    }
};

pub const IndexData = union(IndexType) {
    u16: []u16,
    u32: []u32,
    pub fn indexType(self: IndexData) IndexType {
        return std.meta.activeTag(self);
    }
    pub fn count(self: IndexData) usize {
        return switch (self) {
            inline else => |s| s.len,
        };
    }
    pub fn bytes(self: IndexData) []const u8 {
        return switch (self) {
            inline else => |s| std.mem.sliceAsBytes(s),
        };
    }
};
