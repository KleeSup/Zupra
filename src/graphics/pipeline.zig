const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const zupra = @import("../root.zig");

const graphics = @import("graphics.zig");
const Vertex2D = graphics.Vertex2D;
const Vertex3D = graphics.Vertex3D;
const VertexSkinned3D = graphics.VertexSkinned3D;
const VertexDebug = graphics.VertexDebug;
const IndexType = graphics.IndexType;

/// Named, hashable vertex layouts.
/// The key stays a single enum value rather than a variable-length attr array.
pub const VertexLayout = enum {
    sprite, // Vertex2D
    mesh, // Vertex3D with pos, normal, uv and tangent
    mesh_unlit, // Vertex3D buffer with only pos, normal and uv for unlit scenes.
    /// VertexSkinned3D. The additional joints/weights are used by GPU skinning
    /// variants and deliberately live in a distinct stream layout so static
    /// meshes keep their smaller Vertex3D stride.
    mesh_skinned,
    mesh_skinned_unlit,
    debug, // VertexDebug
    fullscreen, // Vertex2D buffer, only pos+uv consumed (screen-space passes)
    mesh_instanced,

    /// Build the sokol vertex-layout state for this layout. Offsets and stride
    /// are set explicitly so behavior never depends on sokol's auto-layout.
    pub fn state(self: VertexLayout) sg.VertexLayoutState {
        var l = sg.VertexLayoutState{};
        switch (self) {
            .sprite => {
                l.buffers[0].stride = @sizeOf(Vertex2D);
                l.attrs[0] = .{ .offset = @offsetOf(Vertex2D, "pos"), .format = .FLOAT2 };
                l.attrs[1] = .{ .offset = @offsetOf(Vertex2D, "uv"), .format = .FLOAT2 };
                l.attrs[2] = .{ .offset = @offsetOf(Vertex2D, "color"), .format = .UBYTE4N };
            },
            .mesh => {
                l.buffers[0].stride = @sizeOf(Vertex3D);
                l.attrs[0] = .{ .offset = @offsetOf(Vertex3D, "pos"), .format = .FLOAT3 };
                l.attrs[1] = .{ .offset = @offsetOf(Vertex3D, "normal"), .format = .FLOAT3 };
                l.attrs[2] = .{ .offset = @offsetOf(Vertex3D, "uv"), .format = .FLOAT2 };
                l.attrs[3] = .{ .offset = @offsetOf(Vertex3D, "tangent"), .format = .FLOAT4 };
                l.attrs[4] = .{ .offset = @offsetOf(Vertex3D, "uv1"), .format = .FLOAT2 };
            },
            .mesh_unlit => {
                l.buffers[0].stride = @sizeOf(Vertex3D);
                l.attrs[0] = .{ .offset = @offsetOf(Vertex3D, "pos"), .format = .FLOAT3 };
                l.attrs[1] = .{ .offset = @offsetOf(Vertex3D, "uv"), .format = .FLOAT2 };
                l.attrs[2] = .{ .offset = @offsetOf(Vertex3D, "uv1"), .format = .FLOAT2 };
            },
            .mesh_skinned => {
                l.buffers[0].stride = @sizeOf(VertexSkinned3D);
                l.attrs[0] = .{ .offset = @offsetOf(VertexSkinned3D, "pos"), .format = .FLOAT3 };
                l.attrs[1] = .{ .offset = @offsetOf(VertexSkinned3D, "normal"), .format = .FLOAT3 };
                l.attrs[2] = .{ .offset = @offsetOf(VertexSkinned3D, "uv"), .format = .FLOAT2 };
                l.attrs[3] = .{ .offset = @offsetOf(VertexSkinned3D, "tangent"), .format = .FLOAT4 };
                l.attrs[4] = .{ .offset = @offsetOf(VertexSkinned3D, "uv1"), .format = .FLOAT2 };
                l.attrs[5] = .{ .offset = @offsetOf(VertexSkinned3D, "joints"), .format = .USHORT4 };
                l.attrs[6] = .{ .offset = @offsetOf(VertexSkinned3D, "weights"), .format = .FLOAT4 };
            },
            .mesh_skinned_unlit => {
                l.buffers[0].stride = @sizeOf(VertexSkinned3D);
                l.attrs[0] = .{ .offset = @offsetOf(VertexSkinned3D, "pos"), .format = .FLOAT3 };
                l.attrs[1] = .{ .offset = @offsetOf(VertexSkinned3D, "uv"), .format = .FLOAT2 };
                l.attrs[2] = .{ .offset = @offsetOf(VertexSkinned3D, "uv1"), .format = .FLOAT2 };
                l.attrs[3] = .{ .offset = @offsetOf(VertexSkinned3D, "joints"), .format = .USHORT4 };
                l.attrs[4] = .{ .offset = @offsetOf(VertexSkinned3D, "weights"), .format = .FLOAT4 };
            },
            .debug => {
                l.buffers[0].stride = @sizeOf(VertexDebug);
                l.attrs[0] = .{ .offset = @offsetOf(VertexDebug, "pos"), .format = .FLOAT3 };
                l.attrs[1] = .{ .offset = @offsetOf(VertexDebug, "color"), .format = .UBYTE4N };
            },
            .fullscreen => {
                l.buffers[0].stride = @sizeOf(Vertex2D);
                l.attrs[0] = .{ .offset = @offsetOf(Vertex2D, "pos"), .format = .FLOAT2 };
                l.attrs[1] = .{ .offset = @offsetOf(Vertex2D, "uv"), .format = .FLOAT2 };
            },
            .mesh_instanced => {
                // Buffer 0: the ordinary per-vertex stream, identical to .mesh.
                l.buffers[0].stride = @sizeOf(Vertex3D);
                l.attrs[0] = .{ .offset = @offsetOf(Vertex3D, "pos"), .format = .FLOAT3 };
                l.attrs[1] = .{ .offset = @offsetOf(Vertex3D, "normal"), .format = .FLOAT3 };
                l.attrs[2] = .{ .offset = @offsetOf(Vertex3D, "uv"), .format = .FLOAT2 };
                l.attrs[3] = .{ .offset = @offsetOf(Vertex3D, "tangent"), .format = .FLOAT4 };
                l.attrs[4] = .{ .offset = @offsetOf(Vertex3D, "uv1"), .format = .FLOAT2 };

                // Buffer 1: one model matrix per instance, as four FLOAT4 rows.
                // PER_INSTANCE: Advances this buffer once per instance instead of once per vertex, so the same
                // index range is redrawn with a different transform each time.
                l.buffers[1].stride = 16 * @sizeOf(f32);
                l.buffers[1].step_func = .PER_INSTANCE;
                l.attrs[5] = .{ .buffer_index = 1, .offset = 0, .format = .FLOAT4 };
                l.attrs[6] = .{ .buffer_index = 1, .offset = 16, .format = .FLOAT4 };
                l.attrs[7] = .{ .buffer_index = 1, .offset = 32, .format = .FLOAT4 };
                l.attrs[8] = .{ .buffer_index = 1, .offset = 48, .format = .FLOAT4 };
            },
        }
        return l;
    }
};

// --- Blend presets ---

pub const BlendMode = enum {
    none,
    alpha, // straight alpha
    premultiplied, // alpha already multiplied into rgb
    additive,

    fn state(self: BlendMode) sg.BlendState {
        return switch (self) {
            .none => .{ .enabled = false },
            .alpha => .{
                .enabled = true,
                .src_factor_rgb = .SRC_ALPHA,
                .dst_factor_rgb = .ONE_MINUS_SRC_ALPHA,
                .op_rgb = .ADD,
                .src_factor_alpha = .ONE,
                .dst_factor_alpha = .ONE_MINUS_SRC_ALPHA,
                .op_alpha = .ADD,
            },
            .premultiplied => .{
                .enabled = true,
                .src_factor_rgb = .ONE,
                .dst_factor_rgb = .ONE_MINUS_SRC_ALPHA,
                .op_rgb = .ADD,
                .src_factor_alpha = .ONE,
                .dst_factor_alpha = .ONE_MINUS_SRC_ALPHA,
                .op_alpha = .ADD,
            },
            .additive => .{
                .enabled = true,
                .src_factor_rgb = .SRC_ALPHA,
                .dst_factor_rgb = .ONE,
                .op_rgb = .ADD,
                .src_factor_alpha = .ZERO,
                .dst_factor_alpha = .ONE,
                .op_alpha = .ADD,
            },
        };
    }
};

// Pass signature

/// Max color attachments. 4 covers a typical G-buffer (albedo, normal, material and emissive).
pub const max_color_attachments = 4;

/// Describes the render target a pipeline will be used with. For the swapchain,
/// leave formats at `.DEFAULT` and sokol fills them from the swapchain; this is
/// `PassSignature.swapchain`. Offscreen passes specify explicit formats.
pub const PassSignature = struct {
    color_count: u8 = 1,
    color_formats: [max_color_attachments]sg.PixelFormat = @splat(.DEFAULT),
    depth_format: sg.PixelFormat = .DEFAULT,
    sample_count: u8 = 1,

    /// Default swapchain target (1 color + depth, formats resolved by sokol).
    pub fn swapchainPass() PassSignature {
        return .{ .sample_count = @intCast(sokol.app.sampleCount()) };
    }

    pub fn swapchainMsaa(samples: u8) PassSignature {
        return .{ .sample_count = samples };
    }
};

// --- Pipeline key + cache ---

/// The full set of state sokol bakes into an `sg.Pipeline`. Two keys that
/// differ in *any* field require two different pipeline objects. Every field
/// is hashable (enums / ints / bools / arrays, but no floats or pointers), so the
/// struct is usable directly as an AutoHashMap key.
pub const PipelineKey = struct {
    /// sokol shader handle is just `extern struct { id: u32 }` so safe to key on
    /// and reused directly when building the pipeline.
    shader: sg.Shader,
    layout: VertexLayout,
    index_type: IndexType,
    pass: PassSignature,
    primitive: sg.PrimitiveType = .TRIANGLES,
    cull: sg.CullMode = .NONE,
    blend: BlendMode = .none,
    depth_test: bool = false,
    depth_write: bool = false,
    face_winding: sg.FaceWinding = .CCW,

    depth_bias_bits: u32 = 0,
    depth_bias_slope_bits: u32 = 0,
    depth_bias_clamp_bits: u32 = 0,

    color_write_mask: sg.ColorMask = .RGBA,

    /// false = non-indexed draw (sokol index_type = .NONE). Debug lines/fills
    /// and fullscreen post-fx triangles use this, but indexed meshes/sprites = true.
    indexed: bool = true,

    pub fn setDepthBias(self: *PipelineKey, bias: f32, slope_scale: f32, clamp: f32) void {
        self.depth_bias_bits = @bitCast(bias);
        self.depth_bias_slope_bits = @bitCast(slope_scale);
        self.depth_bias_clamp_bits = @bitCast(clamp);
    }
};

/// Maps the framework index tag to sokol's enum.
pub fn toSokolIndexType(it: IndexType) sg.IndexType {
    return switch (it) {
        .u16 => .UINT16,
        .u32 => .UINT32,
    };
}

fn buildDesc(key: PipelineKey) sg.PipelineDesc {
    var desc = sg.PipelineDesc{
        .shader = key.shader,
        .layout = key.layout.state(),
        .index_type = if (key.indexed) toSokolIndexType(key.index_type) else .NONE,
        .primitive_type = key.primitive,
        .cull_mode = key.cull,
        .sample_count = @intCast(key.pass.sample_count),
        .face_winding = key.face_winding,
    };

    desc.depth = .{
        .pixel_format = key.pass.depth_format,
        .compare = if (key.depth_test) .LESS_EQUAL else .ALWAYS,
        .write_enabled = key.depth_write,
        .bias = @bitCast(key.depth_bias_bits),
        .bias_slope_scale = @bitCast(key.depth_bias_slope_bits),
        .bias_clamp = @bitCast(key.depth_bias_clamp_bits),
    };

    desc.color_count = @intCast(key.pass.color_count);
    if (key.pass.color_count == 0) {
        desc.colors[0].pixel_format = .NONE;
    } else {
        const blend = key.blend.state();
        var i: usize = 0;
        while (i < key.pass.color_count and i < max_color_attachments) : (i += 1) {
            desc.colors[i] = .{
                .pixel_format = key.pass.color_formats[i],
                .blend = blend,
                .write_mask = key.color_write_mask,
            };
        }
    }
    return desc;
}

/// Derives and caches `sg.Pipeline` objects keyed by `PipelineKey`. One cache
/// can serve every batcher and pass in the framework.
pub const PipelineCache = struct {
    allocator: std.mem.Allocator,
    map: std.AutoHashMapUnmanaged(PipelineKey, sg.Pipeline) = .{},

    pub fn init(allocator: std.mem.Allocator) PipelineCache {
        return .{ .allocator = allocator };
    }

    /// Returns the cached pipeline for `key`, creating it on first request.
    pub fn get(self: *PipelineCache, key: PipelineKey) !sg.Pipeline {
        const gop = try self.map.getOrPut(self.allocator, key);
        if (!gop.found_existing) {
            gop.value_ptr.* = sg.makePipeline(buildDesc(key));
        }
        return gop.value_ptr.*;
    }

    /// Destroys every cached pipeline and frees the map.
    pub fn deinit(self: *PipelineCache) void {
        var it = self.map.valueIterator();
        while (it.next()) |p| sg.destroyPipeline(p.*);
        self.map.deinit(self.allocator);
    }
};
