const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const zupra = @import("../root.zig");

const graphics = @import("graphics.zig");
const Vertex2D = graphics.Vertex2D;
const Vertex3D = graphics.Vertex3D;
const VertexDebug = graphics.VertexDebug;
const IndexType = graphics.IndexType;

/// Named, hashable vertex layouts.
/// The key stays a single enum value rather than a variable-length attr array.
pub const VertexLayout = enum {
    sprite, // Vertex2D
    mesh, // Vertex3D with pos, normal, uv and tangent
    mesh_unlit, // Vertex3D buffer with only pos, normal and uv for unlit scenes.
    debug, // VertexDebug
    fullscreen, // Vertex2D buffer, only pos+uv consumed (screen-space passes)

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
            },
            .mesh_unlit => {
                l.buffers[0].stride = @sizeOf(Vertex3D);
                l.attrs[0] = .{ .offset = @offsetOf(Vertex3D, "pos"), .format = .FLOAT3 };
                l.attrs[1] = .{ .offset = @offsetOf(Vertex3D, "uv"), .format = .FLOAT2 };
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

/// Max color attachments. 4 covers a typical G-buffer (albedo / normal / material / emissive).
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
        return .{ .sample_count = @intCast(@import("sokol").app.sampleCount()) };
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

    /// false = non-indexed draw (sokol index_type = .NONE). Debug lines/fills
    /// and fullscreen post-fx triangles use this, but indexed meshes/sprites = true.
    indexed: bool = true,
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
    };

    desc.color_count = @intCast(key.pass.color_count);
    const blend = key.blend.state();
    var i: usize = 0;
    while (i < key.pass.color_count and i < max_color_attachments) : (i += 1) {
        desc.colors[i] = .{
            .pixel_format = key.pass.color_formats[i],
            .blend = blend,
        };
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
