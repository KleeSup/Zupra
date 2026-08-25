//! src/render/render_world.zig
//!
//! Retained-mode front end for SceneRenderer.
//!
//! NOT A SECOND RENDERER. RenderWorld owns object LIFETIME and a spatial
//! structure; it then submits through the ordinary immediate API. Every feature
//! added to SceneRenderer works here automatically, and there is no parallel
//! implementation to keep in step -- which is the whole reason to build it this
//! way rather than as a rival path.
//!
//! WHY IT HAS TO EXIST. Three features are blocked on persistence, and none of
//! them can be done from an immediate API that discards its list every frame:
//!
//!   * SPATIAL CULLING. A BVH or grid only pays for itself if it survives across
//!     frames. Rebuilt every frame it costs more than the linear scan it
//!     replaces, so scaling past a few thousand objects needs objects that stay.
//!   * PER-OBJECT MOTION VECTORS. TAA reprojects with camera motion only, so
//!     anything moving under its own transform ghosts. Fixing that needs each
//!     object's PREVIOUS matrix, which has nowhere to live when the list is
//!     thrown away.
//!   * CACHED STATIC SHADOWS. Re-rendering every caster every frame is most of
//!     the shadow cost, and skipping the unchanged ones needs to know which
//!     objects changed -- which needs identity across frames.
//!
//! The immediate API stays useful and stays supported: debug geometry, UI,
//! one-off draws. Filament and Unreal both keep exactly this split.
//!
//! THREADING: single-threaded, like the rest of the renderer. Handles are
//! stable slots with a generation counter, so they survive storage reallocation
//! and detect use-after-remove.

const std = @import("std");
const math = @import("../math.zig");
const culling = @import("culling.zig");
const model_mod = @import("model.zig");
const SceneRenderer = @import("scene.zig").SceneRenderer;

const Matrix = math.Matrix;
const Vec3 = math.Vec3;
const Quaternion = math.Quaternion;
const Model = model_mod.Model;
const ModelInstance = model_mod.ModelInstance;
const Sphere = culling.Sphere;
const Frustum = culling.Frustum;

/// A modest leaf size keeps the tree shallow without making a leaf's exact
/// sphere tests expensive. The renderer still culls per-submesh later; this is
/// only the retained-world broadphase.
const bvh_leaf_size = 8;

/// One node of the retained static-object hierarchy. Internal nodes own two
/// child indices; leaves address a contiguous range in `bvh_indices`.
const BvhNode = struct {
    /// Deliberately a sphere rather than an AABB: it uses the same robust test
    /// as SceneRenderer and remains conservative under arbitrary transforms.
    bounds: Sphere = .empty,
    first: u32 = 0,
    count: u32 = 0,
    left: u32 = 0,
    right: u32 = 0,

    fn isLeaf(self: BvhNode) bool {
        return self.count != 0;
    }
};

/// Stable reference to an object in the world.
///
/// Index plus generation rather than a pointer: appending can reallocate the
/// backing slot array, so callers must not retain internal object addresses.
/// The generation makes a stale handle detectable instead of silently
/// addressing whatever object took the slot.
pub const Handle = struct {
    index: u32,
    generation: u32,

    pub const invalid = Handle{ .index = std.math.maxInt(u32), .generation = 0 };

    pub fn isValid(self: Handle) bool {
        return self.index != Handle.invalid.index;
    }
};

/// How an object is expected to move. This is a promise the caller makes, and
/// the optimisations that depend on it are only correct if it holds.
pub const Mobility = enum(u8) {
    /// Never moves after being added. Its shadow can be cached, its bounds never
    /// refitted, and its spatial cell never revisited. Calling setTransform on a
    /// static object is a bug and is reported.
    static,
    /// Moves occasionally -- doors, platforms, anything driven by events rather
    /// than every frame. Treated as static until it actually changes.
    stationary,
    /// Moves most frames. Never cached, and always contributes motion vectors.
    dynamic,
};

pub const Object = struct {
    model: *Model,
    position: Vec3 = .{ .x = 0, .y = 0, .z = 0 },
    rotation: Quaternion = .{ 0, 0, 0, 1 },
    scale: Vec3 = .{ .x = 1, .y = 1, .z = 1 },

    mobility: Mobility = .dynamic,
    visible: bool = true,
    /// Submitted to the shadow passes. Off for light-fixture geometry, which
    /// otherwise shadows the light inside it -- see ModelInstance.cast_shadows.
    cast_shadows: bool = true,

    // ---- derived, maintained by RenderWorld ----

    /// World matrix, recomputed only when the transform changes.
    matrix: Matrix = undefined,
    /// LAST FRAME's world matrix. The reason a retained world enables motion
    /// vectors at all: this simply has nowhere to live in an immediate API.
    prev_matrix: Matrix = undefined,
    /// World-space bounds, refitted with the matrix.
    bounds: Sphere = .empty,
    /// Set when the transform changed since the last submit. It tells the
    /// motion-vector pass which objects actually moved. Shadow-cache
    /// invalidation is tracked separately at world scope, because changing one
    /// caster invalidates every cached view that might contain it.
    dirty: bool = true,

    /// Transient camera-query marker. A monotonically increasing stamp avoids
    /// clearing a visible flag across every object before each BVH query, while
    /// also ensuring a candidate can be submitted only once.
    camera_query_stamp: u32 = 0,

    generation: u32 = 0,
};

/// Sorts an index range by the centre of its object's bounding sphere while a
/// BVH subtree is built. Keeping indices (rather than Object values) means
/// handles stay stable and the retained objects never move in memory.
const BvhSortContext = struct {
    objects: []const Object,
    indices: []u32,
    axis: u2,

    pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
        const a_index = ctx.indices[a];
        const b_index = ctx.indices[b];
        const av = axisComponent(ctx.objects[a_index].bounds.center, ctx.axis);
        const bv = axisComponent(ctx.objects[b_index].bounds.center, ctx.axis);
        if (av != bv) return av < bv;
        // A deterministic tie break prevents all co-located props from
        // depending on the sort's unspecified order.
        return a_index < b_index;
    }

    pub fn swap(ctx: @This(), a: usize, b: usize) void {
        std.mem.swap(u32, &ctx.indices[a], &ctx.indices[b]);
    }
};

pub const Stats = struct {
    /// Live objects in the world, including explicitly hidden objects.
    total: u32 = 0,
    /// Objects that survived camera culling last submit.
    visible: u32 = 0,
    /// Objects whose transform changed since the previous submit. With cached
    /// shadows this is what the shadow passes would have to redo.
    moved: u32 = 0,
};

pub const RenderWorld = struct {
    allocator: std.mem.Allocator,
    objects: std.ArrayList(Object) = .empty,
    /// Free slots, so removal does not reorder live objects. Reordering would
    /// invalidate every handle pointing past the removed one.
    free_list: std.ArrayList(u32) = .empty,
    /// Bumped on every removal, so a handle to a reused slot fails the
    /// generation check rather than silently addressing its replacement.
    next_generation: u32 = 1,

    /// Rebuild-on-mutation hierarchy for live, visible, stationary geometry.
    /// Dynamic objects are intentionally excluded: rebuilding a static tree
    /// every animation frame would be worse than their short linear walk.
    bvh_nodes: std.ArrayList(BvhNode) = .empty,
    bvh_indices: std.ArrayList(u32) = .empty,
    /// Dynamic objects plus any object without a usable non-empty bound. This
    /// list is rebuilt only when membership changes, never for dynamic motion.
    linear_indices: std.ArrayList(u32) = .empty,
    spatial_dirty: bool = true,
    camera_query_stamp: u32 = 0,

    /// Set by every public mutation that can change the set or pose of shadow
    /// casters. Kept until submit() because SceneRenderer builds its shadow
    /// views in begin(), before it receives this world's submissions.
    shadow_cache_dirty: bool = false,

    stats: Stats = .{},

    pub fn init(allocator: std.mem.Allocator) RenderWorld {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *RenderWorld) void {
        self.objects.deinit(self.allocator);
        self.free_list.deinit(self.allocator);
        self.bvh_nodes.deinit(self.allocator);
        self.bvh_indices.deinit(self.allocator);
        self.linear_indices.deinit(self.allocator);
    }

    pub fn add(self: *RenderWorld, object: Object) !Handle {
        var obj = object;
        obj.matrix = composeMatrix(obj.position, obj.rotation, obj.scale);
        // A new object has no history, so its previous matrix is its current
        // one. Anything else would give it a spurious velocity on its first
        // frame and smear it across the screen.
        obj.prev_matrix = obj.matrix;
        obj.bounds = modelBounds(obj.model).transform(obj.matrix);
        obj.dirty = true;
        // Query stamps are derived state, never caller-authored input. Reset
        // it when reusing a slot so a stale stamp cannot make a new object look
        // camera-visible before the next query marks it.
        obj.camera_query_stamp = 0;

        if (self.free_list.pop()) |index| {
            obj.generation = self.takeGeneration();
            self.objects.items[index] = obj;
            self.invalidateSpatialIndex();
            self.invalidateShadowCache();
            return .{ .index = index, .generation = obj.generation };
        }

        obj.generation = self.takeGeneration();
        const index: u32 = @intCast(self.objects.items.len);
        try self.objects.append(self.allocator, obj);
        self.invalidateSpatialIndex();
        self.invalidateShadowCache();
        return .{ .index = index, .generation = obj.generation };
    }

    pub fn remove(self: *RenderWorld, handle: Handle) void {
        const obj = self.resolveMut(handle) orelse return;
        obj.visible = false;
        // Generation zero marks the slot dead: no live handle carries it, so
        // nothing can resolve here until the slot is reused.
        obj.generation = 0;
        self.free_list.append(self.allocator, handle.index) catch {};
        self.invalidateSpatialIndex();
        self.invalidateShadowCache();
    }

    /// Return a read-only snapshot of a live object. Keeping a pointer into
    /// RenderWorld would be unsafe because a later add() can reallocate its
    /// backing slot array; use the explicit mutators to change world state.
    pub fn get(self: *const RenderWorld, handle: Handle) ?Object {
        const obj = self.resolve(handle) orelse return null;
        return obj.*;
    }

    /// Move an object. Marks it dirty, refits its bounds, and preserves its
    /// previous matrix for motion vectors.
    pub fn setTransform(self: *RenderWorld, handle: Handle, position: Vec3, rotation: Quaternion, scale: Vec3) void {
        const obj = self.resolveMut(handle) orelse return;
        if (std.meta.eql(obj.position, position) and
            @reduce(.And, obj.rotation == rotation) and
            std.meta.eql(obj.scale, scale)) return;
        if (obj.mobility == .static) std.log.warn("RenderWorld: setTransform on a .static object, use .stationary for things that move rarely", .{});

        obj.position = position;
        obj.rotation = rotation;
        obj.scale = scale;
        obj.matrix = composeMatrix(position, rotation, scale);
        obj.bounds = modelBounds(obj.model).transform(obj.matrix);
        obj.dirty = true;
        // A dynamic object remains on the linear path, so its motion needs no
        // tree rebuild. Static/stationary objects change BVH membership/bounds.
        if (obj.mobility != .dynamic) self.invalidateSpatialIndex();
        self.invalidateShadowCache();
    }

    pub fn setPosition(self: *RenderWorld, handle: Handle, position: Vec3) void {
        const obj = self.resolve(handle) orelse return;
        self.setTransform(handle, position, obj.rotation, obj.scale);
    }

    pub fn setVisible(self: *RenderWorld, handle: Handle, visible: bool) void {
        const obj = self.resolveMut(handle) orelse return;
        if (obj.visible == visible) return;
        obj.visible = visible;
        self.invalidateSpatialIndex();
        self.invalidateShadowCache();
    }

    /// Change the spatial/caching contract for an object. Use this rather than
    /// modifying Object.mobility through an out-of-band copy: membership in
    /// the retained BVH and the shadow-cache eligibility both depend on it.
    pub fn setMobility(self: *RenderWorld, handle: Handle, mobility: Mobility) void {
        const obj = self.resolveMut(handle) orelse return;
        if (obj.mobility == mobility) return;
        obj.mobility = mobility;
        // Re-submit one moving frame so a policy change cannot accidentally
        // reuse a cached tile or leave a stale velocity vector behind.
        obj.dirty = true;
        self.invalidateSpatialIndex();
        self.invalidateShadowCache();
    }

    /// Change whether this object contributes depth to shadow maps. Like
    /// visibility, this must invalidate a retained tile: otherwise a disabled
    /// caster leaves its old silhouette behind, or an enabled one never
    /// appears until a light happens to move.
    pub fn setCastShadows(self: *RenderWorld, handle: Handle, cast_shadows: bool) void {
        const obj = self.resolveMut(handle) orelse return;
        if (obj.cast_shadows == cast_shadows) return;
        obj.cast_shadows = cast_shadows;
        self.invalidateShadowCache();
    }

    /// Mark cached shadow depth stale on the next submit.
    ///
    /// Use this escape hatch when a model changes outside RenderWorld's
    /// mutators, for example when its mesh data or material alpha coverage is
    /// replaced in place. Over-invalidation is intentional: a redundant depth
    /// redraw costs a frame, while stale cached depth is a visible correctness
    /// bug.
    pub fn invalidateShadowCache(self: *RenderWorld) void {
        self.shadow_cache_dirty = true;
    }

    /// Refit/rebuild the retained camera-culling index on the next submit.
    ///
    /// Call this if geometry bounds change outside the normal object mutators,
    /// for example after replacing a model's mesh data in place. Pair it with
    /// invalidateShadowCache() when the changed geometry casts shadows.
    pub fn invalidateSpatialIndex(self: *RenderWorld) void {
        self.spatial_dirty = true;
    }

    /// Query the retained hierarchy against the camera and submit the survivors
    /// through the ordinary immediate API.
    ///
    /// This is the entire integration surface: everything below this line is a
    /// normal scene.draw(), so every renderer feature applies unchanged.
    ///
    /// Call between scene.begin() and scene.end().
    pub fn submit(self: *RenderWorld, scene: *SceneRenderer) void {
        // SceneRenderer copies the caller's camera in begin(), fixes its
        // viewport, and applies this frame's TAA jitter. Match that exact
        // projection here: an unjittered broadphase could reject a surface
        // that the renderer's jittered per-submesh culling would draw.
        const frustum = Frustum.fromViewProj(scene.camera.viewProjection());

        // SceneRenderer has already called ShadowRenderer.build() in begin().
        // This companion API invalidates both persistent cache entries and the
        // current frame's cached-view skip flags, so a change reaches the atlas
        // immediately rather than one frame late.
        if (self.shadow_cache_dirty) scene.invalidateCachedShadowsThisFrame();

        self.stats = .{};

        const query_stamp = self.nextCameraQueryStamp();
        if (!scene.frustum_culling) {
            // Respect SceneRenderer's diagnostic toggle exactly: disabling
            // culling means every visible object takes the ordinary main path.
            self.markAllCameraCandidates(frustum, false, query_stamp);
        } else if (self.ensureSpatialIndex()) {
            self.markBvhCandidates(frustum, query_stamp);
            self.markLinearCandidates(frustum, query_stamp);
        } else {
            // Allocation failure during a rebuild is never allowed to make
            // geometry disappear. Keep the dirty index for a later retry and
            // use the original conservative linear query this frame.
            self.markAllCameraCandidates(frustum, true, query_stamp);
        }

        for (self.objects.items) |*obj| {
            if (obj.generation == 0) continue;
            self.stats.total += 1;
            if (obj.dirty) self.stats.moved += 1;
            if (!obj.visible) continue;

            var inst = obj.model.instance();
            inst.position = obj.position;
            inst.rotation = obj.rotation;
            inst.scale = obj.scale;
            inst.cast_shadows = obj.cast_shadows;
            // An idle .dynamic object must never enter a cached shadow tile:
            // it may begin moving after the cache decision but before any
            // caller reports another transform. Stationary/static objects can
            // become cacheable after their one dirty frame has rendered.
            inst.shadow_cacheable = shadowCacheable(obj);

            if (obj.camera_query_stamp != query_stamp) {
                // Camera culling must never become shadow culling. A doorway,
                // truck wheel or wall behind the camera can still cast into the
                // visible part of a light volume, so enqueue its caster only.
                if (obj.cast_shadows) scene.drawShadowOnly(inst);
                continue;
            }

            self.stats.visible += 1;
            if (obj.dirty) {
                scene.drawMoved(inst, obj.prev_matrix);
            } else {
                scene.draw(inst);
            }
        }

        // History rolls forward AFTER submission, so a motion-vector pass reads
        // the same pair of matrices the draw used. This deliberately includes
        // hidden and camera-culled objects: when one re-enters the view, it
        // must not inherit a transform from an arbitrarily old visible frame.
        for (self.objects.items) |*obj| {
            if (obj.generation == 0) continue;
            obj.prev_matrix = obj.matrix;
            obj.dirty = false;
        }
        self.shadow_cache_dirty = false;
    }

    /// Build a new index when the set of static/stationary visible objects
    /// changed. The temporary arrays make this transactional: an out-of-memory
    /// failure leaves the old arrays intact, but `spatial_dirty` prevents them
    /// from being queried as if they described the new world.
    fn rebuildSpatialIndex(self: *RenderWorld) !void {
        var new_nodes: std.ArrayList(BvhNode) = .empty;
        errdefer new_nodes.deinit(self.allocator);
        var new_indices: std.ArrayList(u32) = .empty;
        errdefer new_indices.deinit(self.allocator);
        var new_linear: std.ArrayList(u32) = .empty;
        errdefer new_linear.deinit(self.allocator);

        for (self.objects.items, 0..) |*obj, i| {
            if (obj.generation == 0 or !obj.visible) continue;
            const index: u32 = @intCast(i);
            if (isBvhObject(obj)) {
                try new_indices.append(self.allocator, index);
            } else {
                // Dynamic and empty-bound objects retain the simple linear
                // path. It has no refit cost when a dynamic transform changes.
                try new_linear.append(self.allocator, index);
            }
        }

        if (new_indices.items.len != 0) {
            _ = try self.buildBvhNode(&new_nodes, &new_indices, 0, new_indices.items.len);
        }

        self.bvh_nodes.deinit(self.allocator);
        self.bvh_indices.deinit(self.allocator);
        self.linear_indices.deinit(self.allocator);
        self.bvh_nodes = new_nodes;
        self.bvh_indices = new_indices;
        self.linear_indices = new_linear;
        self.spatial_dirty = false;
    }

    fn ensureSpatialIndex(self: *RenderWorld) bool {
        if (!self.spatial_dirty) return true;
        self.rebuildSpatialIndex() catch return false;
        return true;
    }

    fn buildBvhNode(
        self: *const RenderWorld,
        nodes: *std.ArrayList(BvhNode),
        indices: *std.ArrayList(u32),
        start: usize,
        end: usize,
    ) !u32 {
        std.debug.assert(start < end);

        var bounds: Sphere = .empty;
        var min_center = self.objects.items[indices.items[start]].bounds.center;
        var max_center = min_center;
        for (indices.items[start..end]) |object_index| {
            const object_bounds = self.objects.items[object_index].bounds;
            bounds = Sphere.merge(bounds, object_bounds);
            min_center.x = @min(min_center.x, object_bounds.center.x);
            min_center.y = @min(min_center.y, object_bounds.center.y);
            min_center.z = @min(min_center.z, object_bounds.center.z);
            max_center.x = @max(max_center.x, object_bounds.center.x);
            max_center.y = @max(max_center.y, object_bounds.center.y);
            max_center.z = @max(max_center.z, object_bounds.center.z);
        }

        // Sphere.merge is mathematically conservative. Add a tiny padding to
        // absorb f32 rounding so a broadphase reject can never hide a child on
        // a frustum boundary. Each leaf still tests its exact object sphere.
        bounds = padBvhBounds(bounds);

        const node_index: u32 = @intCast(nodes.items.len);
        try nodes.append(self.allocator, .{ .bounds = bounds });

        const count = end - start;
        if (count <= bvh_leaf_size) {
            nodes.items[node_index] = .{
                .bounds = bounds,
                .first = @intCast(start),
                .count = @intCast(count),
            };
            return node_index;
        }

        const axis = widestAxis(min_center, max_center);
        std.sort.pdqContext(start, end, BvhSortContext{
            .objects = self.objects.items,
            .indices = indices.items,
            .axis = axis,
        });
        const middle = start + count / 2;
        const left = try self.buildBvhNode(nodes, indices, start, middle);
        const right = try self.buildBvhNode(nodes, indices, middle, end);
        nodes.items[node_index] = .{
            .bounds = bounds,
            .left = left,
            .right = right,
        };
        return node_index;
    }

    fn nextCameraQueryStamp(self: *RenderWorld) u32 {
        self.camera_query_stamp +%= 1;
        if (self.camera_query_stamp == 0) {
            // Stamps are only transient. Once the counter wraps, clear them
            // once and resume at one; a 32-bit wrap is decades away in practice.
            for (self.objects.items) |*obj| obj.camera_query_stamp = 0;
            self.camera_query_stamp = 1;
        }
        return self.camera_query_stamp;
    }

    /// Generation zero marks a freed slot, so never hand it to a newly-added
    /// object when the monotonically increasing counter wraps.
    fn takeGeneration(self: *RenderWorld) u32 {
        if (self.next_generation == 0) self.next_generation = 1;
        const generation = self.next_generation;
        self.next_generation +%= 1;
        if (self.next_generation == 0) self.next_generation = 1;
        return generation;
    }

    fn markAllCameraCandidates(self: *RenderWorld, frustum: Frustum, cull: bool, stamp: u32) void {
        for (self.objects.items) |*obj| {
            if (obj.generation == 0 or !obj.visible) continue;
            if (!cull or cameraVisible(frustum, obj.bounds)) obj.camera_query_stamp = stamp;
        }
    }

    fn markBvhCandidates(self: *RenderWorld, frustum: Frustum, stamp: u32) void {
        if (self.bvh_nodes.items.len == 0) return;
        self.markBvhNodeCandidates(0, frustum, stamp);
    }

    fn markBvhNodeCandidates(self: *RenderWorld, node_index: u32, frustum: Frustum, stamp: u32) void {
        const node = self.bvh_nodes.items[node_index];
        if (!frustum.intersectsSphere(node.bounds)) return;

        if (node.isLeaf()) {
            const first: usize = node.first;
            const end = first + node.count;
            for (self.bvh_indices.items[first..end]) |object_index| {
                const obj = &self.objects.items[object_index];
                // Test the original sphere at a leaf, not only the combined
                // node sphere. This makes the BVH query exactly match the
                // camera-frustum rule of the former linear implementation.
                if (isBvhObject(obj) and cameraVisible(frustum, obj.bounds)) {
                    obj.camera_query_stamp = stamp;
                }
            }
            return;
        }

        self.markBvhNodeCandidates(node.left, frustum, stamp);
        self.markBvhNodeCandidates(node.right, frustum, stamp);
    }

    fn markLinearCandidates(self: *RenderWorld, frustum: Frustum, stamp: u32) void {
        for (self.linear_indices.items) |object_index| {
            const obj = &self.objects.items[object_index];
            if (obj.generation == 0 or !obj.visible or isBvhObject(obj)) continue;
            if (cameraVisible(frustum, obj.bounds)) obj.camera_query_stamp = stamp;
        }
    }

    fn resolve(self: *const RenderWorld, handle: Handle) ?*const Object {
        if (!handle.isValid() or handle.index >= self.objects.items.len) return null;
        const obj = &self.objects.items[handle.index];
        if (obj.generation != handle.generation or obj.generation == 0) return null;
        return obj;
    }

    fn resolveMut(self: *RenderWorld, handle: Handle) ?*Object {
        if (!handle.isValid() or handle.index >= self.objects.items.len) return null;
        const obj = &self.objects.items[handle.index];
        if (obj.generation != handle.generation or obj.generation == 0) return null;
        return obj;
    }
};

/// Membership predicate for the static hierarchy. Visibility is included so a
/// hidden object consumes neither a BVH leaf nor a linear camera query; toggles
/// invalidate the index and rebuild it before the next culling pass.
fn isBvhObject(obj: *const Object) bool {
    return obj.generation != 0 and
        obj.visible and
        obj.mobility != .dynamic and
        !obj.bounds.isEmpty();
}

/// Matches SceneRenderer.visible() for the bound-bearing objects that reach the
/// retained query. Empty bounds must stay visible, because the renderer's
/// established policy is to favour a harmless extra draw over disappearing
/// geometry when an importer could not construct a bound.
fn cameraVisible(frustum: Frustum, bounds: Sphere) bool {
    return bounds.isEmpty() or frustum.intersectsSphere(bounds);
}

fn shadowCacheable(obj: *const Object) bool {
    return obj.mobility != .dynamic and !obj.dirty;
}

fn axisComponent(v: Vec3, axis: u2) f32 {
    return switch (axis) {
        0 => v.x,
        1 => v.y,
        else => v.z,
    };
}

fn widestAxis(min_center: Vec3, max_center: Vec3) u2 {
    const x = max_center.x - min_center.x;
    const y = max_center.y - min_center.y;
    const z = max_center.z - min_center.z;
    if (x >= y and x >= z) return 0;
    if (y >= z) return 1;
    return 2;
}

fn padBvhBounds(bounds: Sphere) Sphere {
    if (bounds.isEmpty()) return bounds;
    return .{
        .center = bounds.center,
        .radius = bounds.radius + @max(0.0001, bounds.radius * 0.0001),
    };
}

/// Union of every submesh's object-space bounds.
fn modelBounds(model: *const Model) Sphere {
    var bounds: Sphere = .empty;
    for (model.meshes) |m| bounds = Sphere.merge(bounds, m.bounds);
    return bounds;
}

fn composeMatrix(position: Vec3, rotation: Quaternion, scale: Vec3) Matrix {
    const s = math.zm.scaling(scale.x, scale.y, scale.z);
    const r = math.zm.matFromQuat(rotation);
    const t = math.zm.translation(position.x, position.y, position.z);
    return math.zm.mul(math.zm.mul(s, r), t);
}

test "RenderWorld marks the shadow cache dirty for caster mutations" {
    var world = RenderWorld.init(std.testing.allocator);
    defer world.deinit();

    // An empty model is enough for this ownership/state test: add() accepts it
    // and gives the object an empty, conservatively visible bound without
    // touching any GPU resource.
    var model = Model{
        .meshes = &.{},
        .materials = &.{},
        .mesh_material = &.{},
        .allocator = std.testing.allocator,
    };

    const handle = try world.add(.{ .model = &model });
    try std.testing.expect(world.shadow_cache_dirty);

    world.shadow_cache_dirty = false;
    world.setTransform(
        handle,
        .{ .x = 1, .y = 0, .z = 0 },
        .{ 0, 0, 0, 1 },
        .{ .x = 1, .y = 1, .z = 1 },
    );
    try std.testing.expect(world.shadow_cache_dirty);

    world.shadow_cache_dirty = false;
    world.setVisible(handle, false);
    try std.testing.expect(world.shadow_cache_dirty);
    world.shadow_cache_dirty = false;
    world.setVisible(handle, false);
    try std.testing.expect(!world.shadow_cache_dirty);

    world.setCastShadows(handle, false);
    try std.testing.expect(world.shadow_cache_dirty);

    world.shadow_cache_dirty = false;
    world.invalidateShadowCache();
    try std.testing.expect(world.shadow_cache_dirty);

    world.shadow_cache_dirty = false;
    world.remove(handle);
    try std.testing.expect(world.shadow_cache_dirty);
}

test "RenderWorld never allocates generation zero after counter wrap" {
    var world = RenderWorld.init(std.testing.allocator);
    defer world.deinit();

    world.next_generation = std.math.maxInt(u32);
    try std.testing.expectEqual(std.math.maxInt(u32), world.takeGeneration());
    try std.testing.expectEqual(@as(u32, 1), world.takeGeneration());
}

test "RenderWorld resets caller-supplied derived query state on add" {
    var world = RenderWorld.init(std.testing.allocator);
    defer world.deinit();

    var model = Model{
        .meshes = &.{},
        .materials = &.{},
        .mesh_material = &.{},
        .allocator = std.testing.allocator,
    };

    const handle = try world.add(.{
        .model = &model,
        .camera_query_stamp = 99,
    });
    const snapshot = world.get(handle) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u32, 0), snapshot.camera_query_stamp);
}

test "RenderWorld BVH separates stationary bounds from dynamic and empty objects" {
    var world = RenderWorld.init(std.testing.allocator);
    defer world.deinit();

    var model = Model{
        .meshes = &.{},
        .materials = &.{},
        .mesh_material = &.{},
        .allocator = std.testing.allocator,
    };

    const static_inside = try world.add(.{ .model = &model, .mobility = .static });
    const stationary_outside = try world.add(.{ .model = &model, .mobility = .stationary });
    const dynamic_inside = try world.add(.{ .model = &model, .mobility = .dynamic });
    const static_empty = try world.add(.{ .model = &model, .mobility = .static });

    // The empty test model lets this test set compact, pure synthetic world
    // bounds without creating a GPU mesh. The last object stays empty by
    // design, exercising the conservative linear path.
    world.objects.items[static_inside.index].bounds = .{ .center = .{ .x = 0, .y = 0, .z = 0 }, .radius = 0.25 };
    world.objects.items[stationary_outside.index].bounds = .{ .center = .{ .x = 10, .y = 0, .z = 0 }, .radius = 0.25 };
    world.objects.items[dynamic_inside.index].bounds = .{ .center = .{ .x = 0.5, .y = 0, .z = 0 }, .radius = 0.25 };

    try world.rebuildSpatialIndex();
    try std.testing.expect(!world.spatial_dirty);
    try std.testing.expectEqual(@as(usize, 2), world.bvh_indices.items.len);
    try std.testing.expectEqual(@as(usize, 2), world.linear_indices.items.len);

    // Axis-aligned cube [-2, 2] in the Frustum plane convention. The static
    // and dynamic objects at the origin must survive; the stationary object at
    // x=10 must not. Empty bounds remain visible by policy.
    const frustum = Frustum{ .planes = .{
        .{ 1, 0, 0, 2 },
        .{ -1, 0, 0, 2 },
        .{ 0, 1, 0, 2 },
        .{ 0, -1, 0, 2 },
        .{ 0, 0, 1, 2 },
        .{ 0, 0, -1, 2 },
    } };
    const stamp = world.nextCameraQueryStamp();
    world.markBvhCandidates(frustum, stamp);
    world.markLinearCandidates(frustum, stamp);

    try std.testing.expectEqual(stamp, world.objects.items[static_inside.index].camera_query_stamp);
    try std.testing.expect(world.objects.items[stationary_outside.index].camera_query_stamp != stamp);
    try std.testing.expectEqual(stamp, world.objects.items[dynamic_inside.index].camera_query_stamp);
    try std.testing.expectEqual(stamp, world.objects.items[static_empty.index].camera_query_stamp);

    world.objects.items[dynamic_inside.index].dirty = false;
    world.objects.items[static_inside.index].dirty = false;
    try std.testing.expect(!shadowCacheable(&world.objects.items[dynamic_inside.index]));
    try std.testing.expect(shadowCacheable(&world.objects.items[static_inside.index]));
}
