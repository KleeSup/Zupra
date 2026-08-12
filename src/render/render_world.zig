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
//! indices into a dense array with a generation counter, so they stay valid
//! across insertions and detect use-after-remove.

const std = @import("std");
const math = @import("../math.zig");
const culling = @import("culling.zig");
const model_mod = @import("model.zig");
const SceneRenderer = @import("scene.zig").SceneRenderer;
const Camera3D = @import("camera3d.zig").Camera3D;

const Matrix = math.Matrix;
const Vec3 = math.Vec3;
const Quaternion = math.Quaternion;
const Model = model_mod.Model;
const ModelInstance = model_mod.ModelInstance;
const Sphere = culling.Sphere;
const Frustum = culling.Frustum;

/// Stable reference to an object in the world.
///
/// Index plus generation rather than a pointer: the entry array is dense and
/// reorders on removal, so a pointer would dangle the moment anything else was
/// removed. The generation makes a stale handle detectable instead of silently
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
pub const Mobility = enum {
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
    /// Set when the transform changed since the last submit. Drives shadow
    /// cache invalidation and tells the motion-vector pass which objects
    /// actually moved.
    dirty: bool = true,

    generation: u32 = 0,
};

pub const Stats = struct {
    /// Objects in the world.
    total: u32 = 0,
    /// Objects that survived camera culling last submit.
    visible: u32 = 0,
    /// Objects whose transform changed since the previous submit. With cached
    /// shadows this is what the shadow passes would have to redo.
    moved: u32 = 0,
};

pub const RenderWorld = struct {
    allocator: std.mem.Allocator,
    objects: std.ArrayListUnmanaged(Object) = .empty,
    /// Free slots, so removal does not reorder live objects. Reordering would
    /// invalidate every handle pointing past the removed one.
    free_list: std.ArrayListUnmanaged(u32) = .empty,
    /// Bumped on every removal, so a handle to a reused slot fails the
    /// generation check rather than silently addressing its replacement.
    next_generation: u32 = 1,

    stats: Stats = .{},

    pub fn init(allocator: std.mem.Allocator) RenderWorld {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *RenderWorld) void {
        self.objects.deinit(self.allocator);
        self.free_list.deinit(self.allocator);
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

        if (self.free_list.pop()) |index| {
            obj.generation = self.next_generation;
            self.next_generation +%= 1;
            self.objects.items[index] = obj;
            return .{ .index = index, .generation = obj.generation };
        }

        obj.generation = self.next_generation;
        self.next_generation +%= 1;
        const index: u32 = @intCast(self.objects.items.len);
        try self.objects.append(self.allocator, obj);
        return .{ .index = index, .generation = obj.generation };
    }

    pub fn remove(self: *RenderWorld, handle: Handle) void {
        const obj = self.resolveMut(handle) orelse return;
        obj.visible = false;
        // Generation zero marks the slot dead: no live handle carries it, so
        // nothing can resolve here until the slot is reused.
        obj.generation = 0;
        self.free_list.append(self.allocator, handle.index) catch {};
    }

    pub fn get(self: *const RenderWorld, handle: Handle) ?*const Object {
        return self.resolve(handle);
    }

    /// Move an object. Marks it dirty, refits its bounds, and preserves its
    /// previous matrix for motion vectors.
    pub fn setTransform(self: *RenderWorld, handle: Handle, position: Vec3, rotation: Quaternion, scale: Vec3) void {
        const obj = self.resolveMut(handle) orelse return;
        if (obj.mobility == .static) {
            // Reported rather than silently allowed: caching decisions elsewhere
            // are only correct because static objects really do not move, and a
            // stale cached shadow is far harder to trace back here than a log
            // line at the moment of the mistake.
            std.log.warn("RenderWorld: setTransform on a .static object — use .stationary for things that move rarely", .{});
        }
        obj.position = position;
        obj.rotation = rotation;
        obj.scale = scale;
        obj.matrix = composeMatrix(position, rotation, scale);
        obj.bounds = modelBounds(obj.model).transform(obj.matrix);
        obj.dirty = true;
    }

    pub fn setPosition(self: *RenderWorld, handle: Handle, position: Vec3) void {
        const obj = self.resolve(handle) orelse return;
        self.setTransform(handle, position, obj.rotation, obj.scale);
    }

    pub fn setVisible(self: *RenderWorld, handle: Handle, visible: bool) void {
        const obj = self.resolveMut(handle) orelse return;
        obj.visible = visible;
    }

    /// Cull against the camera and submit the survivors through the ordinary
    /// immediate API.
    ///
    /// This is the entire integration surface: everything below this line is a
    /// normal scene.draw(), so every renderer feature applies unchanged.
    ///
    /// Call between scene.begin() and scene.end().
    pub fn submit(self: *RenderWorld, scene: *SceneRenderer, camera: Camera3D) void {
        const frustum = Frustum.fromViewProj(camera.viewProjection());

        self.stats = .{ .total = @intCast(self.objects.items.len) };

        for (self.objects.items) |*obj| {
            if (obj.generation == 0 or !obj.visible) continue;
            if (obj.dirty) self.stats.moved += 1;

            // Camera culling here rather than in SceneRenderer: an object
            // rejected now never enters any queue, so it costs nothing further.
            // NOTE this is the CAMERA test only -- shadow submission is culled
            // per light view inside the renderer, because an object behind the
            // camera can still cast into view.
            if (!frustum.intersectsSphere(obj.bounds)) continue;
            self.stats.visible += 1;

            var inst = obj.model.instance();
            inst.position = obj.position;
            inst.rotation = obj.rotation;
            inst.scale = obj.scale;
            inst.cast_shadows = obj.cast_shadows;
            if (obj.dirty) {
                scene.drawMoved(inst, obj.prev_matrix);
            } else {
                scene.draw(inst);
            }
        }

        // History rolls forward AFTER submission, so a motion-vector pass reads
        // the same pair of matrices the draw used.
        for (self.objects.items) |*obj| {
            if (obj.generation == 0) continue;
            obj.prev_matrix = obj.matrix;
            obj.dirty = false;
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
