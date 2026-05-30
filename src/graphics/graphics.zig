const std = @import("std");

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
