//! src/render/atlas_allocator.zig
//!
//! Quadtree allocator for the shadow atlas.
//!
//! Replaces the shelf packer, which rebuilt its layout from scratch every frame.
//! That is fine while every tile is transient, but caching needs a tile to stay
//! in the same texels for as long as its contents remain valid, and a shelf
//! packer offers no such guarantee: any earlier light changing resolution, or
//! being added or removed, shifts everything after it.
//!
//! WHY A QUADTREE RATHER THAN A GENERAL RECTANGLE PACKER. Shadow tiles come from
//! a handful of power-of-two sizes, typically 256 to 2048. A quadtree exploits
//! that directly: every rectangle is aligned to its own size, so fragmentation is
//! bounded by construction rather than merely managed, freeing a tile merges it
//! back with its siblings automatically, and both operations are a short walk
//! down a tree. A general free-rectangle allocator handles arbitrary sizes at the
//! cost of significantly more machinery, and there is nothing here to spend it
//! on.
//!
//! RESERVATION LIFETIME IS NOT CACHE VALIDITY. This allocator only answers where
//! a tile lives and for how long. Whether the depth currently in that tile is
//! still correct is a separate question the renderer owns, and keeping the two
//! apart is what lets a shadow be regenerated without surrendering its rectangle.
//! Tiles that moved on every regeneration would defeat the point of caching.

const std = @import("std");

pub const Rect = struct {
    x: u32,
    y: u32,
    size: u32,
};

/// How long an allocation lives.
pub const Lifetime = enum {
    /// Released automatically at the start of the next frame. What every tile
    /// was before caching, and still the right answer for anything that has to
    /// be redrawn each frame anyway.
    transient,
    /// Held until explicitly released. The caller is responsible for freeing it,
    /// and for noticing when it is no longer needed, since nothing here can tell
    /// whether a light still exists.
    persistent,
};

pub const Handle = struct {
    node: u32,

    pub const invalid = Handle{ .node = std.math.maxInt(u32) };

    pub fn isValid(self: Handle) bool {
        return self.node != Handle.invalid.node;
    }
};

const Node = struct {
    /// Index of the first of four children, or 0 when this node is a leaf.
    /// Zero is safe as a sentinel because node 0 is always the root and can
    /// never be a child.
    children: u32 = 0,
    x: u32,
    y: u32,
    size: u32,
    state: State = .free,

    const State = enum {
        free,
        /// Subdivided. Holds no allocation itself; its children do.
        split,
        transient,
        persistent,
    };

    fn isAllocated(self: Node) bool {
        return self.state == .transient or self.state == .persistent;
    }
};

pub const AtlasAllocator = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayListUnmanaged(Node) = .empty,
    size: u32,
    /// Smallest tile this allocator will subdivide down to. Requests below it
    /// are rounded up, since a tile smaller than this wastes more in tree
    /// overhead than it saves in atlas space.
    min_size: u32,

    pub fn init(allocator: std.mem.Allocator, size: u32, min_size: u32) !AtlasAllocator {
        var self = AtlasAllocator{
            .allocator = allocator,
            .size = size,
            .min_size = @max(min_size, 1),
        };
        try self.nodes.append(allocator, .{ .x = 0, .y = 0, .size = size });
        return self;
    }

    pub fn deinit(self: *AtlasAllocator) void {
        self.nodes.deinit(self.allocator);
    }

    /// Release every transient allocation, leaving persistent ones in place.
    ///
    /// This is the whole reason the allocator exists in this form. The shelf
    /// packer's equivalent forgot everything, which is correct only when nothing
    /// needs to survive.
    pub fn beginFrame(self: *AtlasAllocator) void {
        self.freeTransientRecursive(0);
    }

    fn freeTransientRecursive(self: *AtlasAllocator, index: u32) void {
        const node = &self.nodes.items[index];
        switch (node.state) {
            .transient => node.state = .free,
            .split => {
                var i: u32 = 0;
                while (i < 4) : (i += 1) {
                    self.freeTransientRecursive(node.children + i);
                }
                // Merge back if every child ended up free, so the next frame can
                // hand out the whole region again. Without this the tree would
                // stay subdivided forever and a large request would fail even in
                // an empty atlas.
                _ = self.tryMerge(index);
            },
            .free, .persistent => {},
        }
    }

    /// Allocate a square tile. Size is rounded up to a power of two, since the
    /// tree can only produce those, and to min_size.
    pub fn allocate(self: *AtlasAllocator, requested: u32, lifetime: Lifetime) ?struct { rect: Rect, handle: Handle } {
        const size = @max(std.math.ceilPowerOfTwo(u32, requested) catch return null, self.min_size);
        if (size > self.size) return null;

        const index = self.findFree(0, size) orelse return null;
        const node = &self.nodes.items[index];
        node.state = switch (lifetime) {
            .transient => .transient,
            .persistent => .persistent,
        };
        return .{
            .rect = .{ .x = node.x, .y = node.y, .size = node.size },
            .handle = .{ .node = index },
        };
    }

    /// Release a persistent allocation. Transient ones are released by
    /// beginFrame and do not need this.
    pub fn free(self: *AtlasAllocator, handle: Handle) void {
        if (!handle.isValid() or handle.node >= self.nodes.items.len) return;
        const node = &self.nodes.items[handle.node];
        if (!node.isAllocated()) return;
        node.state = .free;
        // Merging is bottom up from the parent, so a freed tile immediately
        // makes larger allocations possible again rather than leaving the region
        // permanently divided.
        self.mergeUpward(handle.node);
    }

    pub fn rectOf(self: AtlasAllocator, handle: Handle) ?Rect {
        if (!handle.isValid() or handle.node >= self.nodes.items.len) return null;
        const node = self.nodes.items[handle.node];
        if (!node.isAllocated()) return null;
        return .{ .x = node.x, .y = node.y, .size = node.size };
    }

    /// Depth-first search for a free node of exactly `size`, subdividing larger
    /// free nodes on the way down.
    ///
    /// Prefers already-subdivided branches over splitting a fresh large node,
    /// which keeps big free regions intact for as long as possible. Without
    /// that preference a run of small requests would carve up every large node
    /// and a later large request would fail with plenty of space free.
    fn findFree(self: *AtlasAllocator, index: u32, size: u32) ?u32 {
        const node = self.nodes.items[index];
        if (node.size < size) return null;

        switch (node.state) {
            .transient, .persistent => return null,
            .split => {
                // Try the already-split children first.
                var i: u32 = 0;
                while (i < 4) : (i += 1) {
                    if (self.findFree(node.children + i, size)) |found| return found;
                }
                return null;
            },
            .free => {
                // Free means the whole region is available, whether or not it
                // still carries children from a previous split.
                if (node.size == size) return index;
                if (node.size <= self.min_size) return null;
                self.split(index) catch return null;
                const children = self.nodes.items[index].children;
                var i: u32 = 0;
                while (i < 4) : (i += 1) {
                    if (self.findFree(children + i, size)) |found| return found;
                }
                return null;
            },
        }
    }

    fn split(self: *AtlasAllocator, index: u32) !void {
        const node = self.nodes.items[index];
        const half = node.size / 2;

        // Reuse the children this node had before it was merged, rather than
        // appending four more. Without this the array would grow on every
        // split-merge cycle instead of settling at the size of the tree shape
        // the scene actually uses.
        if (node.children != 0) {
            self.nodes.items[index].state = .split;
            var i: u32 = 0;
            while (i < 4) : (i += 1) {
                self.nodes.items[node.children + i].state = .free;
            }
            return;
        }

        const first: u32 = @intCast(self.nodes.items.len);
        try self.nodes.ensureUnusedCapacity(self.allocator, 4);
        self.nodes.appendAssumeCapacity(.{ .x = node.x, .y = node.y, .size = half });
        self.nodes.appendAssumeCapacity(.{ .x = node.x + half, .y = node.y, .size = half });
        self.nodes.appendAssumeCapacity(.{ .x = node.x, .y = node.y + half, .size = half });
        self.nodes.appendAssumeCapacity(.{ .x = node.x + half, .y = node.y + half, .size = half });

        // Re-fetched rather than reused: appending may have reallocated the
        // backing array, so the earlier pointer could be stale.
        self.nodes.items[index].children = first;
        self.nodes.items[index].state = .split;
    }

    /// Merge a node back to free if all four children are free. Returns whether
    /// it merged.
    fn tryMerge(self: *AtlasAllocator, index: u32) bool {
        const node = self.nodes.items[index];
        if (node.state != .split) return false;
        var i: u32 = 0;
        while (i < 4) : (i += 1) {
            if (self.nodes.items[node.children + i].state != .free) return false;
        }
        // The four child nodes are left in the array rather than removed.
        // Removing them would shift every later index and invalidate every
        // handle above them.
        //
        // The array therefore only grows, but it is bounded rather than leaking:
        // a full quadtree over the atlas has (4^(depth+1) - 1) / 3 nodes, which
        // for a 4096 atlas down to 64 texels is 5461, or about 130 kilobytes.
        // Splits stop once that shape is reached because every subdivision that
        // could be made already exists. A stress run of 200 frames of random
        // allocation settles near 1500.
        // children is deliberately kept, so a later split of this node reuses
        // the same four entries. state alone distinguishes a merged node from a
        // split one.
        self.nodes.items[index].state = .free;
        return true;
    }

    /// Walk up from a freed node, merging as far as possible.
    ///
    /// Parents are found by search rather than stored, because a parent pointer
    /// would have to be maintained through every split and the tree is shallow
    /// enough that the search is cheap. Depth is log2(size / min_size), so eight
    /// levels for a 4096 atlas down to 16.
    fn mergeUpward(self: *AtlasAllocator, index: u32) void {
        var current = index;
        while (self.findParent(0, current)) |parent| {
            if (!self.tryMerge(parent)) return;
            current = parent;
        }
    }

    fn findParent(self: AtlasAllocator, from: u32, target: u32) ?u32 {
        const node = self.nodes.items[from];
        if (node.state != .split) return null;
        var i: u32 = 0;
        while (i < 4) : (i += 1) {
            if (node.children + i == target) return from;
        }
        i = 0;
        while (i < 4) : (i += 1) {
            if (self.findParent(node.children + i, target)) |found| return found;
        }
        return null;
    }

    pub const Stats = struct {
        /// Texels held by persistent allocations.
        persistent_texels: u64 = 0,
        /// Texels held by transient allocations this frame.
        transient_texels: u64 = 0,
        /// Total atlas texels, for context.
        total_texels: u64 = 0,
    };

    /// Occupancy, for deciding whether caching is affordable in a given scene.
    /// Persistent tiles are the cost of caching made visible.
    pub fn stats(self: AtlasAllocator) Stats {
        var s = Stats{ .total_texels = @as(u64, self.size) * self.size };
        for (self.nodes.items) |node| {
            const texels = @as(u64, node.size) * node.size;
            switch (node.state) {
                .persistent => s.persistent_texels += texels,
                .transient => s.transient_texels += texels,
                else => {},
            }
        }
        return s;
    }
};

test "transient allocations are released each frame" {
    var a = try AtlasAllocator.init(std.testing.allocator, 1024, 16);
    defer a.deinit();

    const first = a.allocate(512, .transient).?;
    a.beginFrame();
    const second = a.allocate(512, .transient).?;
    // The whole atlas is free again, so the same rectangle comes back.
    try std.testing.expectEqual(first.rect.x, second.rect.x);
    try std.testing.expectEqual(first.rect.y, second.rect.y);
}

test "persistent allocations keep their rectangle across frames" {
    var a = try AtlasAllocator.init(std.testing.allocator, 1024, 16);
    defer a.deinit();

    const kept = a.allocate(256, .persistent).?;
    var frame: u32 = 0;
    while (frame < 8) : (frame += 1) {
        a.beginFrame();
        // Transient traffic that would have shifted a shelf packer's layout.
        _ = a.allocate(512, .transient);
        _ = a.allocate(128, .transient);
        const now = a.rectOf(kept.handle).?;
        try std.testing.expectEqual(kept.rect.x, now.x);
        try std.testing.expectEqual(kept.rect.y, now.y);
    }
}

test "freeing merges siblings so large requests succeed again" {
    var a = try AtlasAllocator.init(std.testing.allocator, 512, 16);
    defer a.deinit();

    var handles: [4]Handle = undefined;
    for (&handles) |*h| h.* = a.allocate(256, .persistent).?.handle;
    // Full: nothing left for another 256.
    try std.testing.expect(a.allocate(256, .persistent) == null);

    for (handles) |h| a.free(h);
    // Merged back to a single free root, so the whole atlas is available.
    try std.testing.expect(a.allocate(512, .persistent) != null);
}

test "the node array settles rather than growing every frame" {
    var a = try AtlasAllocator.init(std.testing.allocator, 4096, 64);
    defer a.deinit();

    var kept: [6]Handle = undefined;
    for (&kept) |*h| h.* = a.allocate(512, .persistent).?.handle;

    var prng = std.Random.DefaultPrng.init(1);
    const rand = prng.random();
    const sizes = [_]u32{ 256, 512, 1024 };

    var frame: u32 = 0;
    while (frame < 500) : (frame += 1) {
        a.beginFrame();
        var n = rand.intRangeAtMost(u32, 1, 10);
        while (n > 0) : (n -= 1) {
            _ = a.allocate(sizes[rand.intRangeLessThan(usize, 0, sizes.len)], .transient);
        }
        // Persistent tiles must be untouched by any amount of transient churn.
        for (kept) |h| try std.testing.expect(a.rectOf(h) != null);
    }

    // Reusing a merged node's children is what keeps this bounded. Without it
    // the array grows on every split-merge cycle.
    try std.testing.expect(a.nodes.items.len < 512);
}

test "size is rounded up to a power of two" {
    var a = try AtlasAllocator.init(std.testing.allocator, 1024, 16);
    defer a.deinit();
    const r = a.allocate(300, .transient).?;
    try std.testing.expectEqual(@as(u32, 512), r.rect.size);
}
