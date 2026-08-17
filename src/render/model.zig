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

const math = @import("../math.zig");
const zm = math.zm;
const mesh_mod = @import("mesh.zig");
const Camera3D = @import("camera3d.zig").Camera3D;
const pipeline = @import("../graphics/pipeline.zig");

const Mesh = mesh_mod.Mesh;
const Material = @import("material.zig").Material;
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

    pub fn setShadingModel(self: *Model, shading: ShadingModel) void {
        for (self.materials) |*mesh| {
            mesh.shading = shading;
        }
    }

    pub fn deinit(self: *Model) void {
        for (self.meshes) |m| m.deinit();
        self.allocator.free(self.meshes);
        self.allocator.free(self.materials);
        self.allocator.free(self.mesh_material);
    }

    /// Create a placed instance of this model (identity transform).
    pub fn instance(self: *const Model) ModelInstance {
        return ModelInstance.init(self);
    }
};

pub const ModelInstance = struct {
    model: *const Model,
    position: Vec3 = .{ .x = 0, .y = 0, .z = 0 },
    rotation: Quaternion = .{ 0, 0, 0, 1 }, // identity quaternion
    scale: Vec3 = .{ .x = 1, .y = 1, .z = 1 },

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

    /// Object -> world. Scale, then rotate, then translate (row-vector order:
    /// leftmost applied first). Uploaded directly per the no-transpose rule.
    pub fn modelMatrix(self: ModelInstance) Matrix {
        const s = zm.scaling(self.scale.x, self.scale.y, self.scale.z);
        const r = zm.matFromQuat(self.rotation);
        const t = zm.translation(self.position.x, self.position.y, self.position.z);
        return zm.mul(zm.mul(s, r), t);
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
        const model_matrix = inst.modelMatrix();
        const model = inst.model;
        for (model.meshes, 0..) |submesh, i| {
            const material = model.materials[model.mesh_material[i]];
            self.renderer.draw(submesh, model_matrix, material);
        }
    }

    pub fn end(self: *ModelBatch) void {
        self.renderer.end();
    }
};
