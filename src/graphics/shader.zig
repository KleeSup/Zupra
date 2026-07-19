//! src/graphics/shader.zig
//!
//! ShaderProgram — a thin wrapper over a compiled sokol shader so that user
//! code never touches sokol.gfx directly to create, bind, or destroy a shader.
//!
//! A ShaderProgram also carries its own *bind-slot contract* (where its texture
//! view, sampler, and uniform blocks live). Consumers like SpriteBatch bind
//! against THESE slots rather than assuming the default shader's constants, so
//! a custom shader with different slots is still drop-in.
//!
//! Shaders are precompiled by sokol-shdc into generated modules; you create a
//! program from the generated desc function, e.g.:
//!
//!     const prog = ShaderProgram.init(shaders.sprite.spriteShaderDesc, .{});
//!     defer prog.deinit();
//!
//! No `@import("sokol")` needed in user code, just pass the generated desc fn and
//! (optionally) the generated slot constants (everything else stays internal).

const sg = @import("sokol").gfx;
const pipeline = @import("pipeline.zig");

pub const VertexLayout = pipeline.VertexLayout;

/// Where a shader's resources are bound. Defaults match the common case of
/// everything at `binding = 0` in the .glsl. Override with the generated
/// constants (UB_*, VIEW_*/IMG_*, SMP_*) when a shader differs.
pub const Slots = struct {
    tex_view: u32 = 0,
    sampler: u32 = 0,
    vs_params: u32 = 0,
    /// Set once the SpriteBatch grows a fragment-uniform upload path.
    fs_params: ?u32 = null,
    /// Per-map UV transform block. null = shader doesn't declare it.
    uv_params: ?u32 = null,
};

pub const Options = struct {
    /// Vertex layout the shader's inputs expect. Sprite shaders use `.sprite`.
    layout: VertexLayout = .sprite,
    slots: Slots = .{},
};

pub const ShaderProgram = struct {
    handle: sg.Shader,
    layout: VertexLayout,
    slots: Slots,

    /// Create from a sokol-shdc generated desc function (passed as a value, so
    /// the caller never names a sokol type). The backend is queried internally.
    pub fn init(desc_fn: anytype, opts: Options) ShaderProgram {
        return .{
            .handle = sg.makeShader(desc_fn(sg.queryBackend())),
            .layout = opts.layout,
            .slots = opts.slots,
        };
    }

    /// Adopt an already-created sokol shader handle (advanced/interop).
    pub fn fromHandle(handle: sg.Shader, opts: Options) ShaderProgram {
        return .{ .handle = handle, .layout = opts.layout, .slots = opts.slots };
    }

    /// True once the shader compiled and is ready to use. A failed compile
    /// leaves it in .FAILED.
    pub fn valid(self: ShaderProgram) bool {
        return sg.queryShaderState(self.handle) == .VALID;
    }

    /// Unregister the shader from sokol. Destroy any pipelines that reference
    /// it first (the PipelineCache owns those).
    pub fn deinit(self: ShaderProgram) void {
        sg.destroyShader(self.handle);
    }
};
