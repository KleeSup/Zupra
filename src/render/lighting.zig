//! src/render/lighting.zig
//!
//! Frame lighting state: the LightStore, the ClusterBuilder, and the bindings
//! every shading path needs.
//!
//! This exists so renderers don't each have to know how clustering works. A
//! renderer calls `bind()` with its shader's slot numbers and gets the three
//! views + sampler attached; it calls `params()` for the uniform block. The
//! froxel layout, the y-flip, the slice math - none of that leaks into
//! mesh.zig or deferred.zig.
//!
//! Frame order (SceneRenderer.begin):
//!     lighting.beginFrame(camera, ambient, screen_w, screen_h)
//!       -> store.flush()      re-sort + re-upload if lights changed
//!       -> cluster.build()    assign punctual lights to froxels
//!       -> params cached      one packed uniform block for every path
//!
//! Then any renderer, forward or deferred, binds the same state.

const std = @import("std");
const sg = @import("sokol").gfx;
const light_mod = @import("light.zig");
const cluster_mod = @import("cluster.zig");
const Camera3D = @import("camera3d.zig").Camera3D;
const Color = @import("../root.zig").Color;

const LightStore = light_mod.LightStore;
const Light = light_mod.Light;
const LightHandle = light_mod.LightHandle;
const GlobalLightParams = light_mod.GlobalLightParams;
const ClusterBuilder = cluster_mod.ClusterBuilder;
const ClusterOptions = cluster_mod.ClusterOptions;

/// The three views + sampler a shading path needs for clustered lighting.
/// Slot numbers differ per shader (mesh has IBL bindings ahead of these,
/// lambert doesn't), so the caller supplies its own generated constants.
pub const BindSlots = struct {
    light_data: u32,
    cluster_table: u32,
    cluster_indices: u32,
    sampler: u32,
};

pub const LightingFrame = struct {
    store: LightStore,
    cluster: ClusterBuilder,
    /// NEAREST + clamp. texelFetch ignores filtering, but the sampler object is
    /// still required to form a sampler2D/usampler2D in the shader.
    data_sampler: sg.Sampler,

    /// Packed once per frame in beginFrame, then handed to every path.
    cached: GlobalLightParams = std.mem.zeroes(GlobalLightParams),

    pub fn init(allocator: std.mem.Allocator, opts: ClusterOptions) LightingFrame {
        return .{
            .store = LightStore.init(allocator),
            .cluster = ClusterBuilder.init(allocator, opts),
            .data_sampler = sg.makeSampler(.{
                .min_filter = .NEAREST,
                .mag_filter = .NEAREST,
                .mipmap_filter = .NEAREST,
                .wrap_u = .CLAMP_TO_EDGE,
                .wrap_v = .CLAMP_TO_EDGE,
                .label = "light_data_sampler",
            }),
        };
    }

    pub fn deinit(self: *LightingFrame) void {
        self.store.deinit();
        self.cluster.deinit();
        sg.destroySampler(self.data_sampler);
    }

    // -----------------------------------------------------------------------
    //  Light management - thin pass-through so callers touch one object.
    // -----------------------------------------------------------------------

    pub fn addLight(self: *LightingFrame, light: Light) !LightHandle {
        return self.store.add(light);
    }

    pub fn removeLight(self: *LightingFrame, handle: LightHandle) void {
        self.store.remove(handle);
    }

    pub fn getLight(self: *LightingFrame, handle: LightHandle) ?*Light {
        return self.store.get(handle);
    }

    pub fn clearLights(self: *LightingFrame) void {
        self.store.clear();
    }

    pub fn lightCount(self: *const LightingFrame) usize {
        return self.store.count();
    }

    /// Coarser grids for weaker hardware; finer grids reduce per-froxel light
    /// counts in dense scenes.
    pub fn reconfigureClusters(self: *LightingFrame, opts: ClusterOptions) void {
        self.cluster.reconfigure(opts);
    }

    // -----------------------------------------------------------------------
    //  Frame
    // -----------------------------------------------------------------------

    /// Upload lights (if changed), assign them to froxels, and pack the uniform
    /// block. Call once per frame BEFORE any pass that shades.
    pub fn beginFrame(
        self: *LightingFrame,
        camera: Camera3D,
        ambient: Color,
        screen_w: f32,
        screen_h: f32,
    ) void {
        self.store.flush();
        self.cluster.build(self.store.punctualLights(), camera);

        const grid = self.cluster.gridParams();
        self.cached = self.store.packGlobals(
            ambient,
            camera.position,
            camera.near,
            camera.far,
            screen_w,
            screen_h,
            self.cluster.sliceParams(),
            .{ grid[0], grid[1] },
            self.cluster.capacity,
            sg.queryFeatures().origin_top_left,
        );
    }

    /// The uniform block, for sg.applyUniforms at the shader's UB_light_params.
    pub fn params(self: *const LightingFrame) GlobalLightParams {
        return self.cached;
    }

    /// Attach the clustered-lighting views + sampler to a bindings struct.
    /// Every shading path calls this with its own generated slot numbers.
    pub fn bind(self: *const LightingFrame, bindings: *sg.Bindings, slots: BindSlots) void {
        bindings.views[slots.light_data] = self.store.view;
        bindings.views[slots.cluster_table] = self.cluster.tableView();
        bindings.views[slots.cluster_indices] = self.cluster.indexListView();
        bindings.samplers[slots.sampler] = self.data_sampler;
    }
};
