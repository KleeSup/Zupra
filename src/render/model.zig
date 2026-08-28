//! src/render/model.zig
//!
//! Multi-material models built on Mesh.
//!
//!   * Model: parallel arrays of submeshes + materials, with mesh_material
//!     mapping each submesh to its material (the glTF "primitive" model, and
//!     the same mapping the old engine used). One submesh = one draw call =
//!     one material.
//!   * ModelInstance: a placed instance — pointer to a shared Model plus a
//!     position, a rotation Quaternion, and a scale. Cheap to copy; many
//!     instances share one Model's GPU buffers.
//!   * ModelBatch: begin(cam) / draw(instance) / end(). Thin layer over
//!     MeshRenderer — it computes each instance's model matrix once, then
//!     issues one MeshRenderer.draw per submesh with that submesh's material.

const std = @import("std");
const zupra = @import("../root.zig");
const sg = @import("sokol").gfx;

const math = @import("../math.zig");
const zm = math.zm;
const mesh_mod = @import("mesh.zig");
const Camera3D = @import("camera3d.zig").Camera3D;
const pipeline = @import("../graphics/pipeline.zig");
const skeletal = @import("skeletal.zig");

const Mesh = mesh_mod.Mesh;
const Material = @import("material.zig").Material;
const Texture = @import("../graphics/texture.zig").Texture;
const ShadingModel = @import("material.zig").ShadingModel;
const MeshRenderer = mesh_mod.MeshRenderer;
const PipelineCache = pipeline.PipelineCache;
const PassSignature = pipeline.PassSignature;
const Vec3 = math.Vec3;
const Quaternion = math.Quaternion;
const Matrix = math.Matrix;
const Color = @import("../root.zig").Color;
const Environment = @import("environment.zig").Environment;

pub const Model = struct {
    meshes: []Mesh,
    materials: []Material,
    mesh_material: []usize, // mesh_material[i] = material index for meshes[i]
    allocator: std.mem.Allocator,

    /// GPU resources made specifically while loading this model. They are kept
    /// separate from `materials`: a Material is also used by procedural models
    /// and can reference a texture/sampler owned by the application, a cache,
    /// or the framework defaults. Only a loader that created the resources may
    /// put them here, and Model.deinit destroys each handle exactly once.
    owned_textures: ?[]Texture = null,
    owned_samplers: ?[]sg.Sampler = null,

    /// Retained glTF node, skin and clip data. Kept separate from static mesh
    /// buffers so ordinary procedural/static models retain their existing tiny
    /// representation and lifetime.
    skeletal_asset: ?skeletal.Asset = null,

    /// Convenience: a one-submesh, one-material model that takes ownership of
    /// `mesh`. Frees its small backing arrays on deinit.
    pub fn fromMesh(allocator: std.mem.Allocator, mesh: Mesh, material: Material) !Model {
        const meshes = try allocator.alloc(Mesh, 1);
        meshes[0] = mesh;
        const materials = try allocator.alloc(Material, 1);
        materials[0] = material;
        const mapping = try allocator.alloc(usize, 1);
        mapping[0] = 0;
        return .{ .meshes = meshes, .materials = materials, .mesh_material = mapping, .allocator = allocator };
    }

    /// Take ownership of caller-built arrays. `mesh_material.len` must equal
    /// `meshes.len`, and each entry must index into `materials`.
    pub fn init(allocator: std.mem.Allocator, meshes: []Mesh, materials: []Material, mesh_material: []usize) Model {
        std.debug.assert(meshes.len == mesh_material.len);
        return .{ .meshes = meshes, .materials = materials, .mesh_material = mesh_material, .allocator = allocator };
    }

    /// Like init(), but additionally transfers ownership of GPU resources that
    /// were created for this model. This is intentionally explicit so adding a
    /// Texture to a hand-authored Material never makes Model.deinit destroy it.
    pub fn initWithOwnedResources(
        allocator: std.mem.Allocator,
        meshes: []Mesh,
        materials: []Material,
        mesh_material: []usize,
        owned_textures: ?[]Texture,
        owned_samplers: ?[]sg.Sampler,
    ) Model {
        std.debug.assert(meshes.len == mesh_material.len);
        return .{
            .meshes = meshes,
            .materials = materials,
            .mesh_material = mesh_material,
            .allocator = allocator,
            .owned_textures = owned_textures,
            .owned_samplers = owned_samplers,
        };
    }

    pub fn setShadingModel(self: *Model, shading: ShadingModel) void {
        for (self.materials) |*material| {
            material.shading = shading;
        }
    }

    pub fn setShadingModelRule(self: *Model, rule: *const fn (material: *Material, material_index: usize) ?ShadingModel) void {
        for (self.materials, 0..) |*material, i| {
            const result = rule(material, i) orelse material.shading;
            material.shading = result;
        }
    }

    pub fn deinit(self: *Model) void {
        for (self.meshes) |m| m.deinit();
        self.allocator.free(self.meshes);
        self.allocator.free(self.materials);
        self.allocator.free(self.mesh_material);

        if (self.owned_textures) |textures| {
            for (textures) |texture| texture.deinit();
            self.allocator.free(textures);
            self.owned_textures = null;
        }
        if (self.owned_samplers) |samplers| {
            for (samplers) |sampler| {
                if (sampler.id != 0) sg.destroySampler(sampler);
            }
            self.allocator.free(samplers);
            self.owned_samplers = null;
        }
        if (self.skeletal_asset) |*asset| {
            asset.deinit();
            self.skeletal_asset = null;
        }
    }

    /// Create a placed instance of this model (identity transform).
    pub fn instance(self: *const Model) ModelInstance {
        return ModelInstance.init(self);
    }

    /// Create independent mutable animation state for one ModelInstance. The
    /// resulting Animator owns its palette textures and must outlive every
    /// instance that references it, then be deinitialized by the caller.
    pub fn createAnimator(self: *const Model, allocator: std.mem.Allocator) !skeletal.Animator {
        const asset = if (self.skeletal_asset) |*value| value else return error.ModelHasNoSkeleton;
        return skeletal.Animator.init(allocator, asset);
    }
};

pub const ModelInstance = struct {
    model: *const Model,
    position: Vec3 = .{ .x = 0, .y = 0, .z = 0 },
    rotation: Quaternion = .{ 0, 0, 0, 1 }, // identity quaternion
    scale: Vec3 = .{ .x = 1, .y = 1, .z = 1 },

    /// Optional mutable pose state for a glTF-skinned Model. It is intentionally
    /// external: one immutable Model can be instanced by many independently
    /// animated characters without duplicating mesh buffers or clip data.
    animator: ?*skeletal.Animator = null,

    /// Whether this instance is submitted to the shadow passes.
    ///
    /// Turning it off can be helpful for correcting wrong light projections. An example
    /// case would be light-fixture geometry, where a lamp mesh surrounding its own
    /// point light occludes that light in every direction, so the fixture's
    /// entire range goes black. Emissive signs, glowing crystals and any prop
    /// that reads as a source have the same problem.
    ///
    /// It also covers geometry whose shadow is wrong rather than absent
    /// (i.e. billboards, fake volumetrics, cards standing in for foliage) and thin
    /// clutter whose shadow nobody would miss.
    ///
    /// The instance still renders normally, still receives shadows, and still
    /// appears in the depth prepass. Only shadow SUBMISSION is skipped.
    cast_shadows: bool = true,

    /// Promise that this caster's geometry, transform, and alpha coverage have
    /// not changed since the cached shadow tile was rendered. The safe default
    /// is false: SceneRenderer then invalidates cached shadow tiles for the
    /// current frame. RenderWorld manages this automatically for retained
    /// static/stationary objects; direct callers should normally leave it off.
    shadow_cacheable: bool = false,

    pub fn init(model: *const Model) ModelInstance {
        return .{ .model = model };
    }

    pub fn setPosition(self: *ModelInstance, p: Vec3) void {
        self.position = p;
    }

    pub fn setRotation(self: *ModelInstance, q: Quaternion) void {
        self.rotation = q;
    }

    pub fn setRotationAxisAngle(self: *ModelInstance, axis: Vec3, angle: f32) void {
        self.rotation = zm.quatFromAxisAngle(zm.f32x4(axis.x, axis.y, axis.z, 0), angle);
    }

    pub fn setScale(self: *ModelInstance, s: Vec3) void {
        self.scale = s;
    }

    pub fn setCastShadows(self: *ModelInstance, enabled: bool) void {
        self.cast_shadows = enabled;
    }

    pub fn setAnimator(self: *ModelInstance, animator: ?*skeletal.Animator) void {
        self.animator = animator;
    }

    /// Object -> world. Scale, then rotate, then translate (row-vector order:
    /// leftmost applied first). Uploaded directly per the no-transpose rule.
    pub fn modelMatrix(self: ModelInstance) Matrix {
        const s = zm.scaling(self.scale.x, self.scale.y, self.scale.z);
        const r = zm.matFromQuat(self.rotation);
        const t = zm.translation(self.position.x, self.position.y, self.position.z);
        return zm.mul(zm.mul(s, r), t);
    }

    /// The per-submesh model transform. Static meshes retain the original
    /// external instance matrix. Skinned glTF primitives need their mesh-node
    /// world transform as well: the palette removes that transform internally
    /// (per the glTF skinning equation), so it must be restored here.
    pub fn meshModelMatrix(self: ModelInstance, mesh: Mesh, previous_pose: bool) Matrix {
        return self.meshModelMatrixWithInstance(mesh, self.modelMatrix(), previous_pose);
    }

    /// Variant for temporal submission paths that already retain the prior
    /// external instance transform. The skeletal pose selection remains
    /// independent, so a moving character gets both root and bone velocity.
    pub fn meshModelMatrixWithInstance(self: ModelInstance, mesh: Mesh, instance: Matrix, previous_pose: bool) Matrix {
        if (!mesh.isSkinned()) return instance;

        const asset = self.model.skeletal_asset orelse return instance;
        const node_world = if (self.animator) |animator|
            animator.nodeWorld(mesh.skin_node_index.?, previous_pose) orelse mesh.skin_bind_world
        else
            mesh.skin_bind_world;
        return zm.mul(zm.mul(node_world, asset.asset_to_engine), instance);
    }

    /// Returns the current/previous GPU palette required by a skinned draw.
    /// A skinned primitive without an attached Animator is deliberately not
    /// rendered by the high-level paths rather than being interpreted as a
    /// static Vertex3D buffer with an incompatible stride.
    pub fn skinning(self: ModelInstance, mesh: Mesh) ?skeletal.Binding {
        if (!mesh.isSkinned()) return null;
        const animator = self.animator orelse return null;
        return animator.binding(mesh.skin_palette_index.?);
    }
};

pub const ModelBatch = struct {
    renderer: MeshRenderer,
    env: Environment = .{},

    pub fn init(cache: *PipelineCache) ModelBatch {
        return .{ .renderer = MeshRenderer.init(cache) };
    }

    /// Set the directional light used by subsequent begin() calls.
    pub fn setEnvironment(self: *ModelBatch, env: Environment) void {
        self.env = env;
    }

    pub fn begin(self: *ModelBatch, camera: Camera3D) void {
        self.renderer.begin(camera, self.env);
    }

    pub fn beginEx(self: *ModelBatch, camera: Camera3D, env: Environment, pass: PassSignature) void {
        self.env = env;
        self.renderer.beginEx(camera, env, pass);
    }

    /// Draw every submesh of the instance's model with its mapped material,
    /// all under the instance's transform.
    pub fn draw(self: *ModelBatch, inst: ModelInstance) void {
        const model = inst.model;
        for (model.meshes, 0..) |submesh, i| {
            const material = model.materials[model.mesh_material[i]];
            const matrix = inst.meshModelMatrix(submesh, false);
            if (inst.skinning(submesh)) |skin| {
                self.renderer.drawSkinned(submesh, matrix, material, skin);
            } else {
                self.renderer.draw(submesh, matrix, material);
            }
        }
    }

    pub fn end(self: *ModelBatch) void {
        self.renderer.end();
    }
};
