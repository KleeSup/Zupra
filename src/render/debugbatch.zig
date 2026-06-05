//! src/render/debugbatch.zig
//!
//! Immediate-mode 2D debug drawing: lines, rectangles, circles (outline + fill).
//!
//! Two runs, two topologies: outlines use LINES, fills use TRIANGLES. Each is a
//! separate pipeline (primitive_type is baked into the pipeline), so they flush
//! independently. Both are NON-INDEXED (PipelineKey.indexed = false) because debug
//! geometry is dynamic and small, so an index buffer would be pure overhead.
//! Vertices stream via sg.appendBuffer into one buffer, same model as SpriteBatch.
//!
//! Not for perf-critical drawing. Lines are 1px (driver-dependent).

const std = @import("std");
const sg = @import("sokol").gfx;
const zupra = @import("../root.zig");

const gfx = @import("../graphics/graphics.zig");
const pipeline = @import("../graphics/pipeline.zig");
const cam = @import("camera.zig");
const math = @import("../math.zig");

const shd = @import("shaders").debug;

const VertexDebug = gfx.VertexDebug;
const PipelineCache = pipeline.PipelineCache;
const PipelineKey = pipeline.PipelineKey;
const PassSignature = pipeline.PassSignature;
const BlendMode = pipeline.BlendMode;
const VertexLayout = pipeline.VertexLayout;
const Camera2D = cam.Camera2D;
const Matrix = math.Matrix;
const Vec2 = math.Vec2;
const Rect = math.Rect;
const Color = zupra.Color;

const VsParams = shd.VsParams;

pub const DebugBatchOptions = struct {
    max_line_verts: u32 = 8192,
    max_fill_verts: u32 = 8192,
    /// Segments used to approximate a circle.
    circle_segments: u32 = 32,
};

pub const DebugBatch = struct {
    cache: *PipelineCache,
    shader: gfx.ShaderProgram,
    vbuf: sg.Buffer, // one streaming buffer; line + fill runs append into it

    line_cpu: []VertexDebug,
    fill_cpu: []VertexDebug,
    line_count: u32 = 0,
    fill_count: u32 = 0,
    segments: u32,

    // begin()..end() state
    active: bool = false,
    mvp: Matrix = undefined,
    pass: PassSignature = PassSignature.swapchain,

    pub fn init(allocator: std.mem.Allocator, cache: *PipelineCache, opts: DebugBatchOptions) !DebugBatch {
        const line_cpu = try allocator.alloc(VertexDebug, opts.max_line_verts);
        errdefer allocator.free(line_cpu);
        const fill_cpu = try allocator.alloc(VertexDebug, opts.max_fill_verts);
        errdefer allocator.free(fill_cpu);

        const vbuf = sg.makeBuffer(.{
            .size = (opts.max_line_verts + opts.max_fill_verts) * @sizeOf(VertexDebug),
            .usage = .{ .vertex_buffer = true, .stream_update = true },
        });

        const shader = gfx.ShaderProgram.init(shd.debugShaderDesc, .{
            .layout = .debug,
            .slots = .{ .vs_params = shd.UB_vs_params },
        });

        return .{
            .cache = cache,
            .shader = shader,
            .vbuf = vbuf,
            .line_cpu = line_cpu,
            .fill_cpu = fill_cpu,
            .segments = opts.circle_segments,
        };
    }

    pub fn deinit(self: *DebugBatch, allocator: std.mem.Allocator) void {
        self.shader.deinit();
        sg.destroyBuffer(self.vbuf);
        allocator.free(self.line_cpu);
        allocator.free(self.fill_cpu);
    }

    pub fn begin(self: *DebugBatch, camera: Camera2D) void {
        self.beginEx(camera, PassSignature.swapchain);
    }

    pub fn beginEx(self: *DebugBatch, camera: Camera2D, pass: PassSignature) void {
        std.debug.assert(!self.active);
        self.active = true;
        self.mvp = camera.viewProjection();
        self.pass = pass;
        self.line_count = 0;
        self.fill_count = 0;
    }

    pub fn end(self: *DebugBatch) void {
        std.debug.assert(self.active);
        self.flushLines();
        self.flushFills();
        self.active = false;
    }

    // --- line primitives ---

    pub fn drawLine(self: *DebugBatch, a: Vec2, b: Vec2, color: Color) void {
        self.ensureLines(2);
        const c = zupra.colors.packColor(color);
        self.pushLine(a, c);
        self.pushLine(b, c);
    }

    pub fn drawRect(self: *DebugBatch, rect: Rect, color: Color) void {
        const x0 = rect.x;
        const y0 = rect.y;
        const x1 = rect.x + rect.width;
        const y1 = rect.y + rect.height;
        self.drawLine(.{ .x = x0, .y = y0 }, .{ .x = x1, .y = y0 }, color);
        self.drawLine(.{ .x = x1, .y = y0 }, .{ .x = x1, .y = y1 }, color);
        self.drawLine(.{ .x = x1, .y = y1 }, .{ .x = x0, .y = y1 }, color);
        self.drawLine(.{ .x = x0, .y = y1 }, .{ .x = x0, .y = y0 }, color);
    }

    pub fn drawCircle(self: *DebugBatch, center: Vec2, radius: f32, color: Color) void {
        const n = self.segments;
        const c = zupra.colors.packColor(color);
        self.ensureLines(n * 2);
        const step = std.math.tau / @as(f32, @floatFromInt(n));
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            const a0 = step * @as(f32, @floatFromInt(i));
            const a1 = step * @as(f32, @floatFromInt(i + 1));
            self.pushLine(.{ .x = center.x + @cos(a0) * radius, .y = center.y + @sin(a0) * radius }, c);
            self.pushLine(.{ .x = center.x + @cos(a1) * radius, .y = center.y + @sin(a1) * radius }, c);
        }
    }

    // --- filled primitives ---

    pub fn fillRect(self: *DebugBatch, rect: Rect, color: Color) void {
        self.ensureFills(6);
        const c = zupra.colors.packColor(color);
        const tl = Vec2{ .x = rect.x, .y = rect.y };
        const tr = Vec2{ .x = rect.x + rect.width, .y = rect.y };
        const br = Vec2{ .x = rect.x + rect.width, .y = rect.y + rect.height };
        const bl = Vec2{ .x = rect.x, .y = rect.y + rect.height };
        self.pushFill(tl, c);
        self.pushFill(tr, c);
        self.pushFill(br, c);
        self.pushFill(tl, c);
        self.pushFill(br, c);
        self.pushFill(bl, c);
    }

    pub fn fillCircle(self: *DebugBatch, center: Vec2, radius: f32, color: Color) void {
        const n = self.segments;
        const c = zupra.colors.packColor(color);
        self.ensureFills(n * 3); // triangle fan as separate triangles
        const step = std.math.tau / @as(f32, @floatFromInt(n));
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            const a0 = step * @as(f32, @floatFromInt(i));
            const a1 = step * @as(f32, @floatFromInt(i + 1));
            self.pushFill(center, c);
            self.pushFill(.{ .x = center.x + @cos(a0) * radius, .y = center.y + @sin(a0) * radius }, c);
            self.pushFill(.{ .x = center.x + @cos(a1) * radius, .y = center.y + @sin(a1) * radius }, c);
        }
    }

    // --- internals ---

    inline fn pushLine(self: *DebugBatch, p: Vec2, c: u32) void {
        self.line_cpu[self.line_count] = .{ .pos = .{ p.x, p.y, 0 }, .color = c };
        self.line_count += 1;
    }

    inline fn pushFill(self: *DebugBatch, p: Vec2, c: u32) void {
        self.fill_cpu[self.fill_count] = .{ .pos = .{ p.x, p.y, 0 }, .color = c };
        self.fill_count += 1;
    }

    fn ensureLines(self: *DebugBatch, n: u32) void {
        if (self.line_count + n > self.line_cpu.len) self.flushLines();
    }

    fn ensureFills(self: *DebugBatch, n: u32) void {
        if (self.fill_count + n > self.fill_cpu.len) self.flushFills();
    }

    fn flushLines(self: *DebugBatch) void {
        if (self.line_count == 0) return;
        self.drawRun(self.line_cpu[0..self.line_count], .LINES);
        self.line_count = 0;
    }

    fn flushFills(self: *DebugBatch) void {
        if (self.fill_count == 0) return;
        self.drawRun(self.fill_cpu[0..self.fill_count], .TRIANGLES);
        self.fill_count = 0;
    }

    fn drawRun(self: *DebugBatch, verts: []const VertexDebug, prim: sg.PrimitiveType) void {
        const offset = sg.appendBuffer(self.vbuf, sg.asRange(verts));

        const key = PipelineKey{
            .shader = self.shader.handle,
            .layout = .debug,
            .index_type = .u32, // ignored: indexed = false
            .indexed = false,
            .pass = self.pass,
            .primitive = prim,
            .cull = .NONE,
            .blend = .alpha,
            .depth_test = false,
            .depth_write = false,
        };
        const pip = self.cache.get(key) catch |err| {
            std.log.err("DebugBatch: pipeline cache failed: {}", .{err});
            return;
        };

        var bindings = sg.Bindings{};
        bindings.vertex_buffers[0] = self.vbuf;
        bindings.vertex_buffer_offsets[0] = offset;

        var vs_params = VsParams{ .mvp = @bitCast(self.mvp) };

        sg.applyPipeline(pip);
        sg.applyBindings(bindings);
        sg.applyUniforms(self.shader.slots.vs_params, sg.asRange(&vs_params));
        sg.draw(0, @intCast(verts.len), 1);
    }
};
