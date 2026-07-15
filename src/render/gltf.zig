//! src/render/gltf.zig
//!
//! Complete glTF / GLB loader. Handles the FULL asset: node hierarchy (world
//! transforms composed by cgltf), right-handed→left-handed conversion done once
//! at the transform level, all PBR material maps, embedded + external textures
//! (deduplicated), robust tangents (authored or computed), and a Scene loader
//! that also pulls cameras and KHR_lights_punctual lights.
//!
//! HANDEDNESS. glTF is right-handed (Y-up, +Z toward viewer); the engine is
//! left-handed. Converting between opposite handedness is inherently
//! orientation-reversing, so we (a) apply a root flip F = diag(1,1,-1) as the
//! outermost transform, which negates Z, and (b) reverse triangle winding to
//! compensate for the mirror F introduces (keeping front faces front under
//! CCW+BACK culling). Normals/tangents transform by the inverse-transpose;
//! tangent handedness (w) flips when the transform determinant is negative.
//! This is done PER NODE via the baked world transform — so multi-part models
//! (a watch's dial, a character's limbs) assemble correctly, which per-vertex
//! Z-flipping could not do.

const std = @import("std");
const zmesh = @import("zmesh");
const c = zmesh.io.zcgltf;

const graphics = @import("../graphics/graphics.zig");
const texmod = @import("../graphics/texture.zig");
const model_mod = @import("model.zig");
const material_mod = @import("material.zig");
const mesh_mod = @import("mesh.zig");
const light_mod = @import("light.zig");
const cameramod = @import("camera3d.zig");
const mathz = @import("../math.zig");
const zm = mathz.zm;
const sg = @import("sokol").gfx;

const Vertex3D = graphics.Vertex3D;
const Texture = texmod.Texture;
const Model = model_mod.Model;
const Material = material_mod.Material;
const MapSlot = material_mod.MapSlot;
const Mesh = mesh_mod.Mesh;
const Light = light_mod.Light;
const Camera3D = cameramod.Camera3D;
const Matrix = mathz.Matrix;
const Vec3 = mathz.Vec3;
const zupra = @import("../root.zig");

/// Right-handed (glTF) → left-handed (engine): negate Z. Applied as the
/// outermost transform to every node's world matrix.
fn rhToLh() Matrix {
    return zm.scaling(1, 1, -1);
}

const flipIndices = false;

// ===========================================================================
//  Public API
// ===========================================================================

pub fn loadFile(allocator: std.mem.Allocator, path: [:0]const u8, base_dir: []const u8) !Model {
    const data = try c.parseFile(.{}, path);
    defer c.freeData(data);
    try c.loadBuffers(.{}, data, path);
    const scene = try buildScene(allocator, data, base_dir);
    // Model-only load: drop camera/light arrays.
    allocator.free(scene.cameras);
    allocator.free(scene.lights);
    return scene.model;
}

pub fn loadMemory(allocator: std.mem.Allocator, bytes: []const u8) !Model {
    const data = try c.parse(.{}, bytes);
    defer c.freeData(data);
    try c.loadBuffers(.{}, data, "");
    const scene = try buildScene(allocator, data, "");
    allocator.free(scene.cameras);
    allocator.free(scene.lights);
    return scene.model;
}

/// A loaded glTF scene: geometry + cameras + lights, all in engine space.
/// Lights can be copied into an Environment; a camera can init a Camera3D.
pub const Scene = struct {
    model: Model,
    cameras: []SceneCamera,
    lights: []Light,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Scene) void {
        self.model.deinit();
        self.allocator.free(self.cameras);
        self.allocator.free(self.lights);
    }

    /// Copy this scene's lights into an Environment (clears existing lights).
    pub fn applyLights(self: Scene, env: *@import("environment.zig").Environment) void {
        env.clearLights();
        for (self.lights) |l| env.addLight(l);
    }
};

pub const SceneCamera = struct {
    position: Vec3,
    target: Vec3,
    yfov: f32, // radians
    znear: f32,
    zfar: f32,

    /// Configure a Camera3D from this glTF camera.
    pub fn apply(self: SceneCamera, cam: *Camera3D) void {
        cam.position = self.position;
        cam.target = self.target;
        cam.fov = self.yfov;
        cam.near = self.znear;
        cam.far = self.zfar;
    }
};

pub fn loadSceneFile(allocator: std.mem.Allocator, path: [:0]const u8, base_dir: []const u8) !Scene {
    const data = try c.parseFile(.{}, path);
    defer c.freeData(data);
    try c.loadBuffers(.{}, data, path);
    return buildScene(allocator, data, base_dir);
}

pub fn loadSceneMemory(allocator: std.mem.Allocator, bytes: []const u8) !Scene {
    const data = try c.parse(.{}, bytes);
    defer c.freeData(data);
    try c.loadBuffers(.{}, data, "");
    return buildScene(allocator, data, "");
}

// ===========================================================================
//  Build
// ===========================================================================

const TexCache = std.AutoHashMapUnmanaged(*c.Image, Texture);

/// Sampler cache key: sokol samplers are a limited pool, so dedup identical
/// wrap/filter combinations across all textures in the model.
const SamplerKey = struct {
    wrap_u: sg.Wrap,
    wrap_v: sg.Wrap,
    min: sg.Filter,
    mag: sg.Filter,
    mip: sg.Filter,
};
const SamplerCache = std.AutoHashMapUnmanaged(SamplerKey, sg.Sampler);

const Builder = struct {
    allocator: std.mem.Allocator,
    base_dir: []const u8,
    cache: TexCache = .{},
    samplers: SamplerCache = .{},
    flip: Matrix,

    meshes: std.ArrayListUnmanaged(Mesh) = .empty,
    materials: std.ArrayListUnmanaged(Material) = .empty,
    mapping: std.ArrayListUnmanaged(usize) = .empty,
    cameras: std.ArrayListUnmanaged(SceneCamera) = .empty,
    lights: std.ArrayListUnmanaged(Light) = .empty,
};

fn buildScene(allocator: std.mem.Allocator, data: *c.Data, base_dir: []const u8) !Scene {
    var b = Builder{ .allocator = allocator, .base_dir = base_dir, .flip = rhToLh() };
    defer b.cache.deinit(allocator);
    defer b.samplers.deinit(allocator);

    // Walk the active scene's node hierarchy. cgltf gives us each node's WORLD
    // transform directly (it composes the parent chain), so no manual recursion
    // for transforms is needed — but we still recurse to visit every node.
    const scene = data.scene orelse (if (data.scenes) |s| &s[0] else null);
    if (scene) |sc| {
        if (sc.nodes) |roots| {
            for (roots[0..sc.nodes_count]) |node| try visitNode(&b, node);
        }
    } else if (data.nodes) |nodes| {
        // No scene defined: visit all nodes.
        for (nodes[0..data.nodes_count]) |*n| try visitNode(&b, n);
    }

    const model = Model{
        .meshes = try b.meshes.toOwnedSlice(allocator),
        .materials = try b.materials.toOwnedSlice(allocator),
        .mesh_material = try b.mapping.toOwnedSlice(allocator),
        .allocator = allocator,
    };
    return Scene{
        .model = model,
        .cameras = try b.cameras.toOwnedSlice(allocator),
        .lights = try b.lights.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

fn visitNode(b: *Builder, node: *c.Node) !void {
    // World transform (glTF space) composed by cgltf, then flip to engine space.
    const world_gltf = zm.matFromArr(node.transformWorld());
    const world = zm.mul(world_gltf, b.flip); // engine-space node transform

    if (node.mesh) |mesh| {
        for (mesh.primitives[0..mesh.primitives_count]) |*prim| {
            try emitPrimitive(b, prim, world);
        }
    }
    if (node.camera) |cam| try emitCamera(b, cam, world);
    if (node.light) |light| try emitLight(b, light, world);

    if (node.children) |children| {
        for (children[0..node.children_count]) |child| try visitNode(b, child);
    }
}

// ---------------------------------------------------------------------------
//  Geometry
// ---------------------------------------------------------------------------

fn emitPrimitive(b: *Builder, prim: *c.Primitive, world: Matrix) !void {
    var pos_acc: ?*c.Accessor = null;
    var norm_acc: ?*c.Accessor = null;
    var uv_acc: ?*c.Accessor = null;
    var uv1_acc: ?*c.Accessor = null;
    var tang_acc: ?*c.Accessor = null;
    for (prim.attributes[0..prim.attributes_count]) |*attr| {
        switch (attr.type) {
            .position => pos_acc = attr.data,
            .normal => norm_acc = attr.data,
            // attr.index is the SET number: TEXCOORD_0, TEXCOORD_1, ...
            .texcoord => switch (attr.index) {
                0 => uv_acc = attr.data,
                1 => uv1_acc = attr.data,
                else => {}, // sets 2+ not supported (rare)
            },
            .tangent => tang_acc = attr.data,
            else => {},
        }
    }
    const positions = pos_acc orelse return; // skip non-renderable primitives
    const vcount = positions.count;

    // Normal matrix (inverse-transpose) for directions; determinant sign for
    // tangent handedness under the mirror.
    const normal_mat = zm.transpose(zm.inverse(world));
    const det = zm.determinant(world)[0];
    const w_sign: f32 = if (det < 0) -1 else 1;

    const verts = try b.allocator.alloc(Vertex3D, vcount);
    defer b.allocator.free(verts);

    var i: usize = 0;
    while (i < vcount) : (i += 1) {
        var p: [3]f32 = .{ 0, 0, 0 };
        var n: [3]f32 = .{ 0, 1, 0 };
        var uv: [2]f32 = .{ 0, 0 };
        var uv1: [2]f32 = .{ 0, 0 };
        var t: [4]f32 = .{ 1, 0, 0, 1 };
        _ = positions.readFloat(i, &p);
        if (norm_acc) |a| _ = a.readFloat(i, &n);
        if (uv_acc) |a| _ = a.readFloat(i, &uv);
        if (uv1_acc) |a| _ = a.readFloat(i, &uv1);
        if (tang_acc) |a| _ = a.readFloat(i, &t);

        // Transform to engine space.
        const pe = zm.mul(zm.f32x4(p[0], p[1], p[2], 1), world);
        const ne = zm.normalize3(zm.mul(zm.f32x4(n[0], n[1], n[2], 0), normal_mat));
        const te = zm.normalize3(zm.mul(zm.f32x4(t[0], t[1], t[2], 0), normal_mat));

        verts[i] = .{
            .pos = .{ pe[0], pe[1], pe[2] },
            .normal = .{ ne[0], ne[1], ne[2] },
            .uv = uv,
            .uv1 = uv1,
            .tangent = .{ te[0], te[1], te[2], t[3] * w_sign },
        };
    }

    try appendMesh(b, prim, verts, vcount, tang_acc == null);
}

fn appendMesh(b: *Builder, prim: *c.Primitive, verts: []Vertex3D, vcount: usize, compute_tan: bool) !void {
    const use_u16 = vcount <= 65536;
    if (prim.indices) |indices| {
        const icount = indices.count;
        if (use_u16) {
            const inds = try b.allocator.alloc(u16, icount);
            defer b.allocator.free(inds);
            readIndicesFlipped(u16, indices, inds);
            if (compute_tan) computeTangents(verts, u16, inds);
            try pushPrim(b, prim, verts, .{ .u16 = inds });
        } else {
            const inds = try b.allocator.alloc(u32, icount);
            defer b.allocator.free(inds);
            readIndicesFlipped(u32, indices, inds);
            if (compute_tan) computeTangents(verts, u32, inds);
            try pushPrim(b, prim, verts, .{ .u32 = inds });
        }
    } else {
        // Non-indexed: build a flipped index list (0,2,1, 3,5,4, ...).
        const icount = vcount;
        if (use_u16) {
            const inds = try b.allocator.alloc(u16, icount);
            defer b.allocator.free(inds);
            fillFlippedSeq(u16, inds);
            if (compute_tan) computeTangents(verts, u16, inds);
            try pushPrim(b, prim, verts, .{ .u16 = inds });
        } else {
            const inds = try b.allocator.alloc(u32, icount);
            defer b.allocator.free(inds);
            fillFlippedSeq(u32, inds);
            if (compute_tan) computeTangents(verts, u32, inds);
            try pushPrim(b, prim, verts, .{ .u32 = inds });
        }
    }
}

fn pushPrim(b: *Builder, prim: *c.Primitive, verts: []Vertex3D, idx: graphics.IndexData) !void {
    const mesh = Mesh.init(verts, idx);
    const mat = try buildMaterial(b, prim);
    const slot = b.materials.items.len;
    try b.meshes.append(b.allocator, mesh);
    try b.materials.append(b.allocator, mat);
    try b.mapping.append(b.allocator, slot);
}

/// Read indices, reversing each triangle's winding (2nd/3rd swapped) to
/// compensate for the RH→LH mirror.
fn readIndicesFlipped(comptime I: type, acc: *c.Accessor, out: []I) void {
    var k: usize = 0;
    while (k + 2 < acc.count + 1 and k + 2 < out.len) : (k += 3) {
        if (comptime flipIndices) {
            out[k + 0] = @intCast(acc.readIndex(k + 0));
            out[k + 1] = @intCast(acc.readIndex(k + 2));
            out[k + 2] = @intCast(acc.readIndex(k + 1));
        } else {
            out[k + 0] = @intCast(acc.readIndex(k + 0));
            out[k + 1] = @intCast(acc.readIndex(k + 1));
            out[k + 2] = @intCast(acc.readIndex(k + 2));
        }
    }
}

fn fillFlippedSeq(comptime I: type, out: []I) void {
    var k: usize = 0;
    while (k + 2 < out.len + 1 and k + 2 < out.len) : (k += 3) {
        if (comptime flipIndices) {
            out[k + 0] = @intCast(k + 0);
            out[k + 1] = @intCast(k + 2);
            out[k + 2] = @intCast(k + 1);
        } else {
            out[k + 0] = @intCast(k + 0);
            out[k + 1] = @intCast(k + 1);
            out[k + 2] = @intCast(k + 2);
        }
    }
}

// ---------------------------------------------------------------------------
//  Materials + textures
// ---------------------------------------------------------------------------

fn buildMaterial(b: *Builder, prim: *c.Primitive) !Material {
    var mat = Material{};
    const gm = prim.material orelse return mat;

    mat.alpha_mode = switch (gm.alpha_mode) {
        .@"opaque" => .opaque_,
        .mask => .mask,
        .blend => .blend,
    };
    mat.alpha_cutoff = gm.alpha_cutoff;
    mat.double_sided = gm.double_sided != 0;

    const es: f32 = if (gm.has_emissive_strength != 0) gm.emissive_strength.emissive_strength else 1.0;
    mat.emissive = .{ .r = gm.emissive_factor[0], .g = gm.emissive_factor[1], .b = gm.emissive_factor[2], .a = 1 };
    mat.emissive_strength = es;
    mat.emissive_map = try loadTexView(b, gm.emissive_texture);

    mat.normal_map = try loadTexView(b, gm.normal_texture);
    if (gm.normal_texture.texture != null) mat.normal_scale = gm.normal_texture.scale;

    mat.occlusion_map = try loadTexView(b, gm.occlusion_texture);
    // glTF stores occlusion STRENGTH in the occlusion texture's `scale` field
    // (cgltf reuses TextureView.scale for normal-scale and occlusion-strength).
    if (gm.occlusion_texture.texture != null) {
        mat.occlusion_strength = gm.occlusion_texture.scale;
    }

    if (gm.has_pbr_metallic_roughness != 0) {
        const pbr = gm.pbr_metallic_roughness;
        mat.base_color = .{ .r = pbr.base_color_factor[0], .g = pbr.base_color_factor[1], .b = pbr.base_color_factor[2], .a = pbr.base_color_factor[3] };
        mat.metallic = pbr.metallic_factor;
        mat.roughness = pbr.roughness_factor;
        mat.base_color_map = try loadTexView(b, pbr.base_color_texture);
        mat.metallic_roughness_map = try loadTexView(b, pbr.metallic_roughness_texture);

        setMapUv(&mat, @intFromEnum(MapSlot.base_color), pbr.base_color_texture);
        setMapUv(&mat, @intFromEnum(MapSlot.metallic_roughness), pbr.metallic_roughness_texture);
    }

    // Per-map UV set + KHR_texture_transform. glTF lets EACH map choose its UV
    // set (occlusion on TEXCOORD_1 is common — baked AO gets its own unwrap) and
    // carry its own scale/rotate/offset. Sampling every map with TEXCOORD_0 is
    // what produced the AO-shaped speckle on models like ChronographWatch.
    setMapUv(&mat, @intFromEnum(MapSlot.normal), gm.normal_texture);
    setMapUv(&mat, @intFromEnum(MapSlot.occlusion), gm.occlusion_texture);
    setMapUv(&mat, @intFromEnum(MapSlot.emissive), gm.emissive_texture);

    mat.sampler = resolveSamplerImpl(b, gm) catch null;

    return mat;
}

/// Record a map's UV set (bit in uv_set) and its KHR_texture_transform, with
/// rotation PRE-BAKED into a 2x2 so the shader runs no trig.
///   KHR_texture_transform composes T * R * S:
///     M = [ cos*sx   sin*sy ]   offset = (tx, ty)
///         [-sin*sx   cos*sy ]
fn setMapUv(mat: *Material, comptime idx: comptime_int, view: c.TextureView) void {
    if (view.texture == null) return;

    var set: i32 = view.texcoord;
    var off: [2]f32 = .{ 0, 0 };
    var rot: f32 = 0;
    var scl: [2]f32 = .{ 1, 1 };

    if (view.has_transform != 0) {
        const t = view.transform;
        off = t.offset;
        rot = t.rotation;
        scl = t.scale;
        if (t.has_texcoord != 0) set = t.texcoord; // the extension may override the set
    }

    const cr = @cos(rot);
    const sr = @sin(rot);
    mat.uv_xforms[idx] = .{
        // FIXED: -sr * scl[1] and sr * scl[0]
        .m = .{ cr * scl[0], -sr * scl[1], sr * scl[0], cr * scl[1] },
        .offset = off,
    };
    if (set == 1) mat.uv_set |= (@as(u8, 1) << idx);
}
/// When the glTF texture has no sampler,
// pick the wrap from the base-color transform. A scale > 1 windows into a
// sub-region (needs CLAMP) but otherwise use REPEAT (glTF default / tiling).
fn wrapForAbsentSampler(view: c.TextureView) sg.Wrap {
    if (view.has_transform != 0) {
        // If a transform is present (scale or offset), it's likely a decal.
        // Clamp the edges so the texture doesn't tile across the rest of the geometry.
        return .CLAMP_TO_EDGE;
    }
    return .REPEAT;
}

/// Get-or-create the sokol sampler matching a material's glTF sampler. Uses the
/// first textured slot found (base_color, then the data maps); if none declares
/// a glTF sampler, returns null and the renderer falls back to its default.
fn resolveSamplerImpl(b: *Builder, gm: *c.Material) !?sg.Sampler {
    const empty = c.TextureView{ .texture = null, .texcoord = 0, .scale = 1, .has_transform = 0, .transform = undefined };
    const views = [_]c.TextureView{
        if (gm.has_pbr_metallic_roughness != 0) gm.pbr_metallic_roughness.base_color_texture else empty,
        gm.normal_texture,
        if (gm.has_pbr_metallic_roughness != 0) gm.pbr_metallic_roughness.metallic_roughness_texture else empty,
        gm.occlusion_texture,
        gm.emissive_texture,
    };
    for (views) |v| {
        const t = v.texture orelse continue; // no texture in this slot -> try next

        var key: SamplerKey = undefined;
        if (t.sampler) |gs| {
            // Explicit glTF sampler: honor it exactly.
            const mf = minFilter(gs.min_filter);
            key = .{
                .wrap_u = wrap(gs.wrap_s),
                .wrap_v = wrap(gs.wrap_t),
                .min = mf.min,
                .mag = magFilter(gs.mag_filter),
                .mip = mf.mip,
            };
        } else {

            // NO glTF sampler (very common — exporters omit it). Spec default is
            // REPEAT, but a KHR_texture_transform that scales UVs > 1 windows a
            // sub-region and relies on CLAMP (e.g. the watch logo plates). Decide
            // from the BASE-COLOR transform (the visibly-addressed map), not just
            // this slot. Deliberate, pragmatic deviation from the literal spec
            // default — every engine does some version of this.
            const decide = if (gm.has_pbr_metallic_roughness != 0 and gm.pbr_metallic_roughness.base_color_texture.texture != null)
                gm.pbr_metallic_roughness.base_color_texture
            else
                v;
            const wm = wrapForAbsentSampler(decide);
            key = .{ .wrap_u = wm, .wrap_v = wm, .min = .LINEAR, .mag = .LINEAR, .mip = .LINEAR };
        }

        if (b.samplers.get(key)) |sm| return sm;
        const sm = sg.makeSampler(.{
            .wrap_u = key.wrap_u,
            .wrap_v = key.wrap_v,
            .min_filter = key.min,
            .mag_filter = key.mag,
            .mipmap_filter = key.mip,
        });
        try b.samplers.put(b.allocator, key, sm);
        return sm;
    }
    return null; // no textures at all -> renderer uses its default sampler
}

fn wrap(w: c.WrapMode) sg.Wrap {
    return switch (w) {
        .clamp_to_edge => .CLAMP_TO_EDGE,
        .mirrored_repeat => .MIRRORED_REPEAT,
        .repeat => .REPEAT,
    };
}

fn minFilter(f: c.FilterType) struct { min: sg.Filter, mip: sg.Filter } {
    return switch (f) {
        .nearest => .{ .min = .NEAREST, .mip = .NEAREST },
        .linear => .{ .min = .LINEAR, .mip = .NEAREST },
        .nearest_mipmap_nearest => .{ .min = .NEAREST, .mip = .NEAREST },
        .linear_mipmap_nearest => .{ .min = .LINEAR, .mip = .NEAREST },
        .nearest_mipmap_linear => .{ .min = .NEAREST, .mip = .LINEAR },
        .linear_mipmap_linear => .{ .min = .LINEAR, .mip = .LINEAR },
        .undefined => .{ .min = .LINEAR, .mip = .LINEAR },
    };
}

fn magFilter(f: c.FilterType) sg.Filter {
    return switch (f) {
        .nearest => .NEAREST,
        else => .LINEAR,
    };
}

fn loadTexView(b: *Builder, view: c.TextureView) !?Texture {
    const texture = view.texture orelse return null;
    const image = texture.image orelse return null;
    if (b.cache.get(image)) |t| return t;

    const loaded: ?Texture = blk: {
        if (image.buffer_view) |bv| {
            if (bv.buffer.data) |ptr| {
                const raw: [*]const u8 = @ptrCast(ptr);
                break :blk Texture.initBuffer(raw[bv.offset .. bv.offset + bv.size]) catch null;
            }
        }
        if (image.uri) |uri_c| {
            const uri = std.mem.span(uri_c);
            if (!std.mem.startsWith(u8, uri, "data:")) {
                const full = std.fs.path.join(b.allocator, &.{ b.base_dir, uri }) catch break :blk null;
                defer b.allocator.free(full);
                const bytes = readWholeFile(b.allocator, full) catch break :blk null;
                defer b.allocator.free(bytes);
                break :blk Texture.initBuffer(bytes) catch null;
            }
        }
        break :blk null;
    };
    if (loaded) |t| try b.cache.put(b.allocator, image, t);
    return loaded;
}

fn readWholeFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const io = zupra.getIo();
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    return try reader.interface.allocRemaining(allocator, .limited(64 * 1024 * 1024));
}

// ---------------------------------------------------------------------------
//  Cameras + lights (Scene)
// ---------------------------------------------------------------------------

fn emitCamera(b: *Builder, cam: *c.Camera, world: Matrix) !void {
    if (cam.type != .perspective) return; // orthographic not mapped yet
    const p = cam.data.perspective;
    // Camera sits at the world translation, looks down its local -Z (glTF); in
    // engine space the forward is derived from the transform's basis.
    const eye = zm.mul(zm.f32x4(0, 0, 0, 1), world);
    const fwd = zm.normalize3(zm.mul(zm.f32x4(0, 0, -1, 0), world));
    try b.cameras.append(b.allocator, .{
        .position = .{ .x = eye[0], .y = eye[1], .z = eye[2] },
        .target = .{ .x = eye[0] + fwd[0], .y = eye[1] + fwd[1], .z = eye[2] + fwd[2] },
        .yfov = p.yfov,
        .znear = p.znear,
        .zfar = if (p.has_zfar != 0) p.zfar else 1000.0,
    });
}

fn emitLight(b: *Builder, light: *c.Light, world: Matrix) !void {
    const eye = zm.mul(zm.f32x4(0, 0, 0, 1), world);
    const dir = zm.normalize3(zm.mul(zm.f32x4(0, 0, -1, 0), world)); // glTF lights aim down -Z
    const pos = Vec3{ .x = eye[0], .y = eye[1], .z = eye[2] };
    const direction = Vec3{ .x = dir[0], .y = dir[1], .z = dir[2] };
    const color = zupra.Color{ .r = light.color[0], .g = light.color[1], .b = light.color[2], .a = 1 };
    const range = if (light.range > 0) light.range else 20.0;

    const l: Light = switch (light.type) {
        .directional => Light.directional(direction, color, light.intensity),
        .point => Light.point(pos, color, light.intensity, range),
        .spot => Light.spot(pos, direction, color, light.intensity, range, std.math.radiansToDegrees(light.spot_inner_cone_angle), std.math.radiansToDegrees(light.spot_outer_cone_angle)),
        .invalid => return,
    };
    try b.lights.append(b.allocator, l);
}

// ---------------------------------------------------------------------------
//  Tangent fallback (Lengyel), for primitives without TANGENT.
// ---------------------------------------------------------------------------

fn computeTangents(verts: []Vertex3D, comptime I: type, indices: []const I) void {
    const A = std.heap.page_allocator;
    const tan = A.alloc([3]f32, verts.len) catch return;
    defer A.free(tan);
    const bit = A.alloc([3]f32, verts.len) catch return;
    defer A.free(bit);
    for (tan) |*t| t.* = .{ 0, 0, 0 };
    for (bit) |*x| x.* = .{ 0, 0, 0 };

    var i: usize = 0;
    while (i + 2 < indices.len) : (i += 3) {
        const a = indices[i];
        const bb = indices[i + 1];
        const cc = indices[i + 2];
        const p0 = verts[a].pos;
        const p1 = verts[bb].pos;
        const p2 = verts[cc].pos;
        const _u0 = verts[a].uv;
        const _u1 = verts[bb].uv;
        const _u2 = verts[cc].uv;
        const e1 = [3]f32{ p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
        const e2 = [3]f32{ p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };
        const du1 = [2]f32{ _u1[0] - _u0[0], _u1[1] - _u0[1] };
        const du2 = [2]f32{ _u2[0] - _u0[0], _u2[1] - _u0[1] };
        const denom = du1[0] * du2[1] - du2[0] * du1[1];
        const f = if (@abs(denom) < 1e-8) 0.0 else 1.0 / denom;
        const t = [3]f32{ f * (du2[1] * e1[0] - du1[1] * e2[0]), f * (du2[1] * e1[1] - du1[1] * e2[1]), f * (du2[1] * e1[2] - du1[1] * e2[2]) };
        const bt = [3]f32{ f * (du1[0] * e2[0] - du2[0] * e1[0]), f * (du1[0] * e2[1] - du2[0] * e1[1]), f * (du1[0] * e2[2] - du2[0] * e1[2]) };
        for ([_]I{ a, bb, cc }) |vi| {
            tan[vi] = .{ tan[vi][0] + t[0], tan[vi][1] + t[1], tan[vi][2] + t[2] };
            bit[vi] = .{ bit[vi][0] + bt[0], bit[vi][1] + bt[1], bit[vi][2] + bt[2] };
        }
    }
    for (verts, 0..) |*v, vi| {
        const n = v.normal;
        var t = tan[vi];
        const ndt = n[0] * t[0] + n[1] * t[1] + n[2] * t[2];
        t = .{ t[0] - n[0] * ndt, t[1] - n[1] * ndt, t[2] - n[2] * ndt };
        const len = @sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
        if (len > 1e-6) {
            t = .{ t[0] / len, t[1] / len, t[2] / len };
        } else {
            t = .{ 1, 0, 0 };
        }
        const cx = n[1] * t[2] - n[2] * t[1];
        const cy = n[2] * t[0] - n[0] * t[2];
        const cz = n[0] * t[1] - n[1] * t[0];
        const w: f32 = if (cx * bit[vi][0] + cy * bit[vi][1] + cz * bit[vi][2] < 0) -1 else 1;
        v.tangent = .{ t[0], t[1], t[2], w };
    }
}
