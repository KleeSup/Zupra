//! Skeletal animation data and GPU palette upload.
//!
//! glTF skinning is intentionally represented as a float texture rather than a
//! large vertex uniform array. A texture avoids the low WebGL2 uniform limit,
//! keeps one shader path across desktop and WebGPU, and has no practical joint
//! cap beyond the device's ordinary 2D texture limit.

const std = @import("std");
const sg = @import("sokol").gfx;
const math = @import("../math.zig");

const zm = math.zm;
const Matrix = math.Matrix;
const Quaternion = math.Quaternion;

/// Number of joint matrices packed across one RGBA32F palette row. Each matrix
/// occupies four texels, so the texture width is 1024 texels.
pub const matrices_per_row: usize = 256;
pub const palette_texture_width: usize = matrices_per_row * 4;

/// These binding slots are intentionally outside every built-in material
/// shader's texture range. Keeping them stable lets the current, G-buffer,
/// depth, shadow and velocity variants share one compact draw binding contract.
pub const palette_view_slot: usize = 13;
pub const previous_palette_view_slot: usize = 14;
pub const palette_sampler_slot: usize = 4;

pub const Transform = struct {
    translation: [3]f32 = .{ 0, 0, 0 },
    rotation: [4]f32 = .{ 0, 0, 0, 1 },
    scale: [3]f32 = .{ 1, 1, 1 },

    pub fn matrix(self: Transform) Matrix {
        const s = zm.scaling(self.scale[0], self.scale[1], self.scale[2]);
        const r = zm.matFromQuat(@bitCast(self.rotation));
        const t = zm.translation(self.translation[0], self.translation[1], self.translation[2]);
        return zm.mul(zm.mul(s, r), t);
    }
};

/// One glTF node retained for skin/animation evaluation. `rest_matrix` preserves
/// matrix-authored static nodes exactly; animated glTF nodes normally use TRS.
pub const Node = struct {
    parent: ?u32,
    rest: Transform,
    rest_matrix: Matrix,
    uses_trs: bool,
};

pub const Skin = struct {
    joints: []u32,
    inverse_bind: []Matrix,
    skeleton_root: ?u32,

    pub fn deinit(self: *Skin, allocator: std.mem.Allocator) void {
        allocator.free(self.joints);
        allocator.free(self.inverse_bind);
    }
};

/// A glTF skin attached to one mesh node. glTF permits the same Skin definition
/// to be referenced by multiple mesh nodes, but each node needs a distinct
/// palette because the skin matrix is relative to that mesh node's world space.
pub const SkinInstance = struct {
    skin_index: u32,
    mesh_node: u32,
};

pub const Interpolation = enum {
    linear,
    step,
    cubic_spline,
};

pub const Path = enum {
    translation,
    rotation,
    scale,

    pub fn componentCount(self: Path) usize {
        return switch (self) {
            .translation, .scale => 3,
            .rotation => 4,
        };
    }
};

/// Owns fully unpacked glTF samples. Keeping times/values compact and flat makes
/// sampling allocation-free, while preserving STEP, LINEAR and CUBICSPLINE data.
pub const Channel = struct {
    node: u32,
    path: Path,
    interpolation: Interpolation,
    times: []f32,
    values: []f32,

    pub fn deinit(self: *Channel, allocator: std.mem.Allocator) void {
        allocator.free(self.times);
        allocator.free(self.values);
    }
};

pub const Clip = struct {
    /// Owned UTF-8 copied from the glTF while cgltf's data is still alive.
    name: ?[]u8,
    channels: []Channel,
    duration: f32,

    pub fn deinit(self: *Clip, allocator: std.mem.Allocator) void {
        for (self.channels) |*channel| channel.deinit(allocator);
        allocator.free(self.channels);
        if (self.name) |name| allocator.free(name);
    }
};

/// Loader-owned skeletal data attached to a Model. All matrices remain in the
/// glTF asset basis; `asset_to_engine` performs the right-handed -> left-handed
/// conversion only when a mesh is drawn, exactly like static glTF loading does.
pub const Asset = struct {
    allocator: std.mem.Allocator,
    nodes: []Node,
    skins: []Skin,
    skin_instances: []SkinInstance,
    clips: []Clip,
    asset_to_engine: Matrix,

    pub fn deinit(self: *Asset) void {
        self.allocator.free(self.nodes);
        for (self.skins) |*skin| skin.deinit(self.allocator);
        self.allocator.free(self.skins);
        self.allocator.free(self.skin_instances);
        for (self.clips) |*clip| clip.deinit(self.allocator);
        self.allocator.free(self.clips);
    }
};

/// Resources a skinned draw binds into its vertex shader. Current and previous
/// palette views are both carried here so the velocity pass can follow bone
/// motion instead of only the model transform.
pub const Binding = struct {
    palette: sg.View,
    previous_palette: sg.View,
    sampler: sg.Sampler,
};

const Palette = struct {
    current_image: sg.Image = .{},
    current_view: sg.View = .{},
    previous_image: sg.Image = .{},
    previous_view: sg.View = .{},
    rows: usize = 0,
    staging: []@Vector(4, f32) = &.{},
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator, joint_count: usize) !Palette {
        const info = sg.queryPixelformat(.RGBA32F);
        if (!info.sample) return error.UnsupportedSkinPaletteFormat;

        const rows = @max(1, (joint_count + matrices_per_row - 1) / matrices_per_row);
        const staging = try allocator.alloc(@Vector(4, f32), rows * palette_texture_width);
        errdefer allocator.free(staging);

        const desc = sg.ImageDesc{
            .width = @intCast(palette_texture_width),
            .height = @intCast(rows),
            .pixel_format = .RGBA32F,
            .usage = .{ .stream_update = true },
            .label = "skinning_palette",
        };
        const current_image = sg.makeImage(desc);
        errdefer sg.destroyImage(current_image);
        const current_view = sg.makeView(.{ .texture = .{ .image = current_image } });
        errdefer sg.destroyView(current_view);
        const previous_image = sg.makeImage(desc);
        errdefer sg.destroyImage(previous_image);
        const previous_view = sg.makeView(.{ .texture = .{ .image = previous_image } });
        errdefer sg.destroyView(previous_view);

        return .{
            .current_image = current_image,
            .current_view = current_view,
            .previous_image = previous_image,
            .previous_view = previous_view,
            .rows = rows,
            .staging = staging,
            .allocator = allocator,
        };
    }

    fn deinit(self: *Palette) void {
        sg.destroyView(self.previous_view);
        sg.destroyImage(self.previous_image);
        sg.destroyView(self.current_view);
        sg.destroyImage(self.current_image);
        self.allocator.free(self.staging);
    }

    fn upload(self: *Palette, image: sg.Image, matrices: []const Matrix) void {
        @memset(self.staging, @Vector(4, f32){ 0, 0, 0, 0 });
        for (matrices, 0..) |matrix, i| {
            const row = i / matrices_per_row;
            const col = (i % matrices_per_row) * 4;
            const base = row * palette_texture_width + col;
            const rows: [4][4]f32 = @bitCast(matrix);
            self.staging[base + 0] = rows[0];
            self.staging[base + 1] = rows[1];
            self.staging[base + 2] = rows[2];
            self.staging[base + 3] = rows[3];
        }
        var data = sg.ImageData{};
        data.mip_levels[0] = sg.asRange(self.staging);
        sg.updateImage(image, data);
    }
};

const PosePalette = struct {
    matrices: []Matrix,
    previous_matrices: []Matrix,
    gpu: Palette,

    fn deinit(self: *PosePalette, allocator: std.mem.Allocator) void {
        self.gpu.deinit();
        allocator.free(self.previous_matrices);
        allocator.free(self.matrices);
    }
};

fn initPosePalette(allocator: std.mem.Allocator, joint_count: usize) !PosePalette {
    const matrices = try allocator.alloc(Matrix, joint_count);
    errdefer allocator.free(matrices);
    const previous_matrices = try allocator.alloc(Matrix, joint_count);
    errdefer allocator.free(previous_matrices);
    const gpu = try Palette.init(allocator, joint_count);
    return .{
        .matrices = matrices,
        .previous_matrices = previous_matrices,
        .gpu = gpu,
    };
}

/// Mutable per-instance state for a Model's skeletal asset. It is deliberately
/// separate from ModelInstance so Models remain immutable/shareable and a model
/// can be animated independently by many instances.
pub const Animator = struct {
    allocator: std.mem.Allocator,
    asset: *const Asset,
    pose: []Transform,
    animated_mask: []u8,
    world: []Matrix,
    previous_world: []Matrix,
    resolve_state: []u8,
    palettes: []PosePalette,
    sampler: sg.Sampler,

    clip_index: ?usize = null,
    time: f32 = 0,
    speed: f32 = 1,
    looping: bool = true,
    playing: bool = false,

    pub fn init(allocator: std.mem.Allocator, asset: *const Asset) !Animator {
        try validateAsset(asset);
        const pose = try allocator.alloc(Transform, asset.nodes.len);
        errdefer allocator.free(pose);
        const animated_mask = try allocator.alloc(u8, asset.nodes.len);
        errdefer allocator.free(animated_mask);
        const world = try allocator.alloc(Matrix, asset.nodes.len);
        errdefer allocator.free(world);
        const previous_world = try allocator.alloc(Matrix, asset.nodes.len);
        errdefer allocator.free(previous_world);
        const resolve_state = try allocator.alloc(u8, asset.nodes.len);
        errdefer allocator.free(resolve_state);
        const palettes = try allocator.alloc(PosePalette, asset.skin_instances.len);
        errdefer allocator.free(palettes);

        var palette_count: usize = 0;
        errdefer {
            for (palettes[0..palette_count]) |*palette| palette.deinit(allocator);
        }
        for (asset.skin_instances) |skin_instance| {
            const skin = asset.skins[skin_instance.skin_index];
            palettes[palette_count] = try initPosePalette(allocator, skin.joints.len);
            palette_count += 1;
        }

        const sampler = sg.makeSampler(.{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
            .mipmap_filter = .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
            .label = "skinning_palette_sampler",
        });
        errdefer sg.destroySampler(sampler);

        var self = Animator{
            .allocator = allocator,
            .asset = asset,
            .pose = pose,
            .animated_mask = animated_mask,
            .world = world,
            .previous_world = previous_world,
            .resolve_state = resolve_state,
            .palettes = palettes,
            .sampler = sampler,
        };
        self.resetPose();
        self.rebuildWorld();
        @memcpy(self.previous_world, self.world);
        self.rebuildPalettes();
        for (self.palettes) |*palette| {
            @memcpy(palette.previous_matrices, palette.matrices);
            palette.gpu.upload(palette.gpu.current_image, palette.matrices);
            palette.gpu.upload(palette.gpu.previous_image, palette.previous_matrices);
        }
        return self;
    }

    pub fn deinit(self: *Animator) void {
        for (self.palettes) |*palette| palette.deinit(self.allocator);
        self.allocator.free(self.palettes);
        sg.destroySampler(self.sampler);
        self.allocator.free(self.resolve_state);
        self.allocator.free(self.previous_world);
        self.allocator.free(self.world);
        self.allocator.free(self.animated_mask);
        self.allocator.free(self.pose);
    }

    pub fn clipCount(self: Animator) usize {
        return self.asset.clips.len;
    }

    pub fn play(self: *Animator, clip_index: usize, looping: bool) void {
        if (clip_index >= self.asset.clips.len) return;
        self.clip_index = clip_index;
        self.time = 0;
        self.looping = looping;
        self.playing = true;
        self.sampleCurrentClip();
        self.rebuildWorld();
        self.rebuildPalettes();
        self.synchronizePrevious();
    }

    pub fn stop(self: *Animator) void {
        self.playing = false;
    }

    pub fn setTime(self: *Animator, time: f32) void {
        self.time = @max(0, time);
        self.sampleCurrentClip();
        self.rebuildWorld();
        self.rebuildPalettes();
        self.synchronizePrevious();
    }

    /// Advance one frame. The pre-update world/palette remains available to the
    /// velocity pass, while the new pose is written to the current palette.
    pub fn update(self: *Animator, delta_seconds: f32) void {
        @memcpy(self.previous_world, self.world);
        for (self.palettes) |*palette| @memcpy(palette.previous_matrices, palette.matrices);

        if (self.playing) {
            if (self.clip_index) |index| {
                const clip = self.asset.clips[index];
                self.time += delta_seconds * self.speed;
                if (clip.duration > 0) {
                    if (self.looping) {
                        self.time = @mod(@max(0, self.time), clip.duration);
                    } else if (self.time >= clip.duration) {
                        self.time = clip.duration;
                        self.playing = false;
                    }
                }
            }
        }

        self.sampleCurrentClip();
        self.rebuildWorld();
        self.rebuildPalettes();
        for (self.palettes) |*palette| {
            palette.gpu.upload(palette.gpu.previous_image, palette.previous_matrices);
            palette.gpu.upload(palette.gpu.current_image, palette.matrices);
        }
    }

    pub fn binding(self: *const Animator, palette_index: u32) ?Binding {
        if (palette_index >= self.palettes.len) return null;
        const palette = self.palettes[palette_index];
        return .{
            .palette = palette.gpu.current_view,
            .previous_palette = palette.gpu.previous_view,
            .sampler = self.sampler,
        };
    }

    pub fn nodeWorld(self: *const Animator, node_index: u32, previous: bool) ?Matrix {
        if (node_index >= self.world.len) return null;
        return if (previous) self.previous_world[node_index] else self.world[node_index];
    }

    /// Start a newly selected/seeked clip without generating a one-frame motion
    /// streak from its old pose. Ordinary `update` calls preserve the prior pose
    /// instead, so TAA receives true bone velocity.
    fn synchronizePrevious(self: *Animator) void {
        @memcpy(self.previous_world, self.world);
        for (self.palettes) |*palette| {
            @memcpy(palette.previous_matrices, palette.matrices);
            palette.gpu.upload(palette.gpu.previous_image, palette.previous_matrices);
            palette.gpu.upload(palette.gpu.current_image, palette.matrices);
        }
    }

    fn resetPose(self: *Animator) void {
        for (self.asset.nodes, 0..) |node, i| self.pose[i] = node.rest;
        @memset(self.animated_mask, 0);
    }

    fn sampleCurrentClip(self: *Animator) void {
        self.resetPose();
        const index = self.clip_index orelse return;
        if (index >= self.asset.clips.len) return;
        const clip = self.asset.clips[index];
        for (clip.channels) |channel| self.sampleChannel(channel);
    }

    fn sampleChannel(self: *Animator, channel: Channel) void {
        if (channel.node >= self.pose.len or channel.times.len == 0) return;
        const key = sampleKey(channel.times, self.time);
        const components = channel.path.componentCount();
        const a = channelValue(channel, key.left, components);
        var value: [4]f32 = a;

        if (key.right != key.left and channel.interpolation != .step) {
            const b = channelValue(channel, key.right, components);
            value = switch (channel.interpolation) {
                .linear => if (channel.path == .rotation) slerp4(a, b, key.t) else lerp4(a, b, key.t),
                .cubic_spline => cubicValue(channel, key.left, key.right, key.t, key.delta, components),
                .step => a,
            };
        }

        switch (channel.path) {
            .translation => {
                self.pose[channel.node].translation = .{ value[0], value[1], value[2] };
                self.animated_mask[channel.node] |= 1;
            },
            .rotation => {
                const q: Quaternion = .{ value[0], value[1], value[2], value[3] };
                self.pose[channel.node].rotation = @bitCast(normalizeQuat(q));
                self.animated_mask[channel.node] |= 2;
            },
            .scale => {
                self.pose[channel.node].scale = .{ value[0], value[1], value[2] };
                self.animated_mask[channel.node] |= 4;
            },
        }
    }

    fn rebuildWorld(self: *Animator) void {
        @memset(self.resolve_state, 0);
        for (self.world, 0..) |_, i| self.resolveWorld(i);
    }

    fn resolveWorld(self: *Animator, index: usize) void {
        if (self.resolve_state[index] == 2) return;
        if (self.resolve_state[index] == 1) {
            // glTF disallows cycles. Failing closed to a local transform keeps a
            // malformed asset from recursing indefinitely or corrupting memory.
            self.world[index] = localMatrix(self.asset.nodes[index], self.pose[index], self.animated_mask[index]);
            self.resolve_state[index] = 2;
            return;
        }

        self.resolve_state[index] = 1;
        const local = localMatrix(self.asset.nodes[index], self.pose[index], self.animated_mask[index]);
        if (self.asset.nodes[index].parent) |parent| {
            if (parent < self.world.len) {
                self.resolveWorld(parent);
                self.world[index] = zm.mul(local, self.world[parent]);
            } else {
                self.world[index] = local;
            }
        } else {
            self.world[index] = local;
        }
        self.resolve_state[index] = 2;
    }

    fn rebuildPalettes(self: *Animator) void {
        for (self.asset.skin_instances, 0..) |skin_instance, palette_index| {
            const skin = self.asset.skins[skin_instance.skin_index];
            const mesh_world = self.world[skin_instance.mesh_node];
            // A zero-scale mesh node has no inverse. It is not a useful skin
            // transform, but retaining identity here avoids NaNs propagating
            // through an otherwise valid scene; its zero draw transform still
            // makes it visually absent as authored.
            const determinant = zm.determinant(mesh_world)[0];
            const inverse_mesh_world = if (@abs(determinant) > 1e-8) zm.inverse(mesh_world) else zm.identity();
            const palette = &self.palettes[palette_index];
            for (skin.joints, 0..) |joint_node, joint_index| {
                // Row-vector equivalent of glTF's
                // inverse(meshWorld) * jointWorld * inverseBindMatrix.
                // Shader matrices are uploaded transposed, so this order gives
                // the standard column-vector result in GLSL.
                palette.matrices[joint_index] = skinPaletteMatrix(skin.inverse_bind[joint_index], self.world[joint_node], inverse_mesh_world);
            }
        }
    }
};

fn skinPaletteMatrix(inverse_bind: Matrix, joint_world: Matrix, inverse_mesh_world: Matrix) Matrix {
    return zm.mul(zm.mul(inverse_bind, joint_world), inverse_mesh_world);
}

const SampleKey = struct {
    left: usize,
    right: usize,
    t: f32,
    delta: f32,
};

fn sampleKey(times: []const f32, time: f32) SampleKey {
    if (times.len == 1 or time <= times[0]) return .{ .left = 0, .right = 0, .t = 0, .delta = 0 };
    const last = times.len - 1;
    if (time >= times[last]) return .{ .left = last, .right = last, .t = 0, .delta = 0 };

    var left: usize = 0;
    var right: usize = last;
    while (right - left > 1) {
        const middle = left + (right - left) / 2;
        if (time < times[middle]) right = middle else left = middle;
    }
    const delta = @max(times[right] - times[left], 1e-8);
    return .{ .left = left, .right = right, .t = std.math.clamp((time - times[left]) / delta, 0, 1), .delta = delta };
}

fn channelValue(channel: Channel, key_index: usize, components: usize) [4]f32 {
    const sample_index = if (channel.interpolation == .cubic_spline) key_index * 3 + 1 else key_index;
    const base = sample_index * components;
    var out = [4]f32{ 0, 0, 0, 1 };
    var i: usize = 0;
    while (i < components) : (i += 1) out[i] = channel.values[base + i];
    return out;
}

fn cubicValue(channel: Channel, left: usize, right: usize, t: f32, delta: f32, components: usize) [4]f32 {
    const p0 = channelValue(channel, left, components);
    const p1 = channelValue(channel, right, components);
    const out_tangent_base = (left * 3 + 2) * components;
    const in_tangent_base = (right * 3 + 0) * components;
    const t2 = t * t;
    const t3 = t2 * t;
    const h00 = 2 * t3 - 3 * t2 + 1;
    const h10 = t3 - 2 * t2 + t;
    const h01 = -2 * t3 + 3 * t2;
    const h11 = t3 - t2;
    var out = [4]f32{ 0, 0, 0, 1 };
    var i: usize = 0;
    while (i < components) : (i += 1) {
        out[i] = h00 * p0[i] +
            h10 * channel.values[out_tangent_base + i] * delta +
            h01 * p1[i] +
            h11 * channel.values[in_tangent_base + i] * delta;
    }
    return out;
}

fn lerp4(a: [4]f32, b: [4]f32, t: f32) [4]f32 {
    return .{
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    };
}

/// glTF specifies spherical interpolation for LINEAR quaternion channels.
/// Choosing the shortest hemisphere also prevents a valid q/-q key pair from
/// needlessly rotating almost a full turn.
fn slerp4(a: [4]f32, b: [4]f32, t: f32) [4]f32 {
    var end = b;
    var dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if (dot < 0) {
        dot = -dot;
        end = .{ -b[0], -b[1], -b[2], -b[3] };
    }
    if (dot > 0.9995) {
        const q: Quaternion = @bitCast(lerp4(a, end, t));
        return @bitCast(normalizeQuat(q));
    }

    const theta = std.math.acos(std.math.clamp(dot, -1.0, 1.0));
    const sin_theta = @sin(theta);
    if (@abs(sin_theta) <= 1e-6) return a;
    const wa = @sin((1.0 - t) * theta) / sin_theta;
    const wb = @sin(t * theta) / sin_theta;
    const q: Quaternion = @bitCast([4]f32{
        a[0] * wa + end[0] * wb,
        a[1] * wa + end[1] * wb,
        a[2] * wa + end[2] * wb,
        a[3] * wa + end[3] * wb,
    });
    return @bitCast(normalizeQuat(q));
}

fn normalizeQuat(q: Quaternion) Quaternion {
    const len2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if (len2 <= 1e-12) return .{ 0, 0, 0, 1 };
    const inv = 1.0 / @sqrt(len2);
    return .{ q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv };
}

fn localMatrix(node: Node, pose: Transform, animated_mask: u8) Matrix {
    return if (node.uses_trs or animated_mask != 0) pose.matrix() else node.rest_matrix;
}

fn validateAsset(asset: *const Asset) !void {
    for (asset.nodes) |node| {
        if (node.parent) |parent| if (parent >= asset.nodes.len) return error.InvalidSkeletalNode;
    }
    for (asset.skins) |skin| {
        if (skin.joints.len != skin.inverse_bind.len) return error.InvalidSkin;
        for (skin.joints) |joint| if (joint >= asset.nodes.len) return error.InvalidSkin;
        if (skin.skeleton_root) |root| if (root >= asset.nodes.len) return error.InvalidSkin;
    }
    for (asset.skin_instances) |instance| {
        if (instance.skin_index >= asset.skins.len or instance.mesh_node >= asset.nodes.len) return error.InvalidSkinInstance;
    }
    for (asset.clips) |clip| {
        for (clip.channels) |channel| {
            if (channel.node >= asset.nodes.len or channel.times.len == 0) return error.InvalidAnimationChannel;
            const components = channel.path.componentCount();
            const sample_count = if (channel.interpolation == .cubic_spline) channel.times.len * 3 else channel.times.len;
            if (channel.values.len != sample_count * components) return error.InvalidAnimationChannel;
            for (channel.times[1..], channel.times[0 .. channel.times.len - 1]) |next, previous| {
                if (next < previous) return error.InvalidAnimationChannel;
            }
        }
    }
}

test "skeletal linear and cubic animation sampling" {
    var times = [_]f32{ 0, 1 };
    var values = [_]f32{ 0, 0, 0, 2, 4, 6 };
    const linear = Channel{
        .node = 0,
        .path = .translation,
        .interpolation = .linear,
        .times = &times,
        .values = &values,
    };
    const key = sampleKey(&times, 0.25);
    const a = channelValue(linear, key.left, 3);
    const b = channelValue(linear, key.right, 3);
    const v = lerp4(a, b, key.t);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), v[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), v[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), v[2], 0.0001);
}

test "skin palette cancels mesh-node space before the draw model restores it" {
    const inverse_bind = zm.identity();
    const mesh_world = zm.translation(2, 0, 0);
    const joint_world = zm.translation(3, 0, 0);
    const palette = skinPaletteMatrix(inverse_bind, joint_world, zm.inverse(mesh_world));
    const final = zm.mul(palette, mesh_world);
    const point = zm.mul(zm.f32x4(0, 0, 0, 1), final);
    try std.testing.expectApproxEqAbs(@as(f32, 3), point[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), point[1], 0.0001);
}

test "slerp chooses the shortest quaternion arc" {
    const half = slerp4(.{ 0, 0, 0, 1 }, .{ 0, 0, 1, 0 }, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, std.math.sqrt1_2), @abs(half[2]), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, std.math.sqrt1_2), @abs(half[3]), 0.0001);
}
