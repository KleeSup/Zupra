//! src/render/shadow.zig
//!
//! Shadow foundation: a single depth ATLAS subdivided into tiles, plus the
//! caster/settings types every light kind renders through.
//!
//! ARCHITECTURE — why an atlas from the very first directional light:
//!
//! One depth texture, subdivided into rectangular tiles by a simple allocator.
//! Everything is a client of this:
//!   * directional single map  -> 1 tile
//!   * cascaded directional     -> N tiles
//!   * spot light               -> 1 tile
//!   * point light              -> 6 tiles (cube faces)
//!   * cached static shadow      -> a tile whose contents persist across frames
//!   * priority/resolution scaling -> larger or smaller tiles per light
//!
//! Per-light textures would make every one of those a special case or a rewrite.
//! A tile allocator makes them all the same operation, which is why AAA shadow
//! systems are atlas-based. The allocator here is deliberately simple (a shelf
//! packer over power-of-two tiles); it can be replaced with a quadtree later
//! without touching any client, because clients only ever ask for a tile and
//! get back a rect + the matrices to render into it.
//!
//! The shader side (pbr_lib.glsl.inc shadowFactor) only ever sees "a light-space
//! matrix and an atlas rect", so directional/spot/point all share ONE fragment
//! path. Hardware comparison sampling (SamplerDesc.compare) does the PCF depth
//! test, which is both correct and cheap.

const std = @import("std");
const culling = @import("culling.zig");
const atlas_alloc = @import("atlas_allocator.zig");
const sg = @import("sokol").gfx;
const math = @import("../math.zig");
const Matrix = math.Matrix;

/// A tile's location within the atlas, in NORMALIZED [0,1] atlas UV. The shader
/// maps a light-space position into [0,1] then scales+biases into this rect, so
/// the sampled texel lands in the right tile.
pub const AtlasRect = extern struct {
    /// x, y, width, height in [0,1] atlas UV.
    uv: [4]f32 = .{ 0, 0, 0, 0 },

    pub fn valid(self: AtlasRect) bool {
        return self.uv[2] > 0 and self.uv[3] > 0;
    }
};

/// One depth-only render: render all shadow casters through `view_proj` into the
/// atlas region `rect`. This is the uniform unit of shadow work — the renderer
/// loops over casters and never branches on light type.
pub const ShadowCaster = struct {
    view_proj: Matrix,
    rect: AtlasRect,
    /// Tile origin + size in TEXELS, for setting the viewport/scissor so this
    /// caster only writes its own tile.
    tile_x: u32,
    tile_y: u32,
    tile_size: u32,
    /// Depth bias applied in the depth-only pass (slope-scaled, in the pipeline).
    depth_bias: f32 = 2.0,
    depth_bias_slope: f32 = 3.0,
    /// Face culling for this caster's depth pass. See ShadowSettings.caster_cull.
    cull: sg.CullMode = .NONE,
    /// This view's frustum, for rejecting casters that cannot affect its tile.
    /// Derived from view_proj rather than stored per light type, so a cascade's
    /// orthographic box, a spot's cone and a cube face all cull through exactly
    /// one code path.
    frustum: culling.Frustum = .{},
};

/// Per-light shadow configuration. Defaults are conservative (low cost); a user
/// dials them up per light.
pub const ShadowSettings = struct {
    enabled: bool = false,
    /// Tile resolution in texels (per cascade / per cube face). 1024 is a modest
    /// default; 2048+ for a hero light.
    resolution: u32 = 1024,
    /// Constant depth bias (texels) to fight shadow acne.
    depth_bias: f32 = 2.0,
    /// Slope-scaled depth bias — surfaces at grazing angles need more.
    slope_bias: f32 = 3.0,
    /// Normal-offset bias, in SHADOW TEXELS rather than world units: push the
    /// sample point along the surface normal to avoid acne without the
    /// peter-panning heavy depth bias causes.
    ///
    /// Texels, not world units, because the whole point of cascades is that a
    /// texel covers wildly different amounts of world space in each one -- a
    /// near cascade might be centimetres per texel where the far one is tens of
    /// centimetres. A fixed world offset tuned for the far cascade is then
    /// grossly oversized in the near one (visible light leaking under contact
    /// shadows), and one tuned for the near cascade leaves the far one full of
    /// acne. Expressed in texels it converts to world units per cascade and
    /// stays correct in all of them.
    normal_bias_texels: f32 = 1.5,
    /// Fraction of max_distance over which shadows fade out at the far end of
    /// their range. 0 disables the fade and restores a hard cutoff.
    ///
    /// Without it, shadows simply stop at max_distance and the boundary reads as
    /// a clean arc across the ground -- more obvious than the cascade seam,
    /// because it is a step from full shadow to none rather than a change in
    /// resolution. Fading to fully lit hides where the shadow range ends, which
    /// is what lets max_distance be tuned for performance instead of for how
    /// visible its edge is.
    ///
    /// Directional lights only. A spot or point light's own attenuation has
    /// already taken its contribution to nothing by the time its shadow range
    /// runs out, so there is no edge there to hide.
    distance_fade: f32 = 0.2,
    /// Width of the cross-fade between adjacent cascades, as a fraction of the
    /// cascade's far split. Without it the switch is instantaneous and shows up
    /// as a hard arc across the ground where resolution and bias change. Costs a
    /// second shadow lookup for fragments inside the band only.
    cascade_blend: f32 = 0.1,
    /// Max distance from the camera the shadow is rendered to. Directional
    /// shadows fit their ortho box to this; beyond it, surfaces are unshadowed
    /// (or faded — see distance fade, a later feature).
    max_distance: f32 = 60.0,
    /// Directional cascade count. 1 = single map (Phase 1). 2-4 = CSM (Phase 2).
    /// The atlas allocates this many tiles for the light.
    cascade_count: u32 = 1,
    /// PCF filter radius in texels. 0 = hard (1 tap), 1 = 3x3, 2 = 5x5.
    ///
    /// Cost is (2r+1)^2 dependent atlas fetches PER SHADOWED LIGHT touching the
    /// fragment, so this is the sharpest dial available when shadowed lights
    /// overlap: three lights at radius 1 is 27 fetches a pixel.
    pcf_radius: u32 = 1,
    /// Face culling for the depth pass. Neither option is right for every scene,
    /// which is why it is a setting rather than a constant:
    ///
    ///   .FRONT records the far side of each occluder, which pushes the recorded
    ///     surface away from the receiver and suppresses acne on curved
    ///     geometry. But where an object MEETS its receiver -- a post standing
    ///     on a floor -- the far surface is the object's underside, coplanar
    ///     with the floor, and any bias then leaks a bright outline around the
    ///     contact.
    ///   .NONE records the nearest surface, so contact is correct and the
    ///     outline disappears, at the cost of more self-shadowing acne that the
    ///     bias settings then have to absorb.
    ///   .BACK is rarely useful here; it is the shading-pass convention.
    ///
    /// .NONE is the default because a wrong-looking contact edge reads as a bug
    /// to everyone, while mild acne reads as texture.
    caster_cull: sg.CullMode = .NONE,
    /// How far behind the fitted volume the near plane is pulled back, in world
    /// units, so geometry BETWEEN the light and the visible slice still gets
    /// into the map. Without it a tall object just outside the cascade is
    /// clipped away and casts nothing into the region the camera can see. Costs
    /// depth precision, so it wants to be roughly the tallest expected caster
    /// rather than an arbitrarily large number.
    caster_extrusion: f32 = 30.0,
};

pub const AtlasOptions = struct {
    /// Atlas edge length in texels. 4096 gives, e.g., 4x 2048 cascade tiles, or
    /// 16x 1024 tiles, or a mix. Configurable; default is a balance of quality
    /// and the ~64MB (D32F 4096^2) footprint.
    size: u32 = 4096,
};

/// Depth atlas + shelf tile allocator. Allocation is per-frame: reset(), then
/// each light requests the tiles it needs; the layout is rebuilt every frame,
/// which keeps it simple and makes priority/resolution changes free. Caching
/// (persisting a tile across frames) is layered on top later by NOT resetting
/// certain tiles — the allocator is designed to allow that without change.
pub const ShadowAtlas = struct {
    size: u32,
    depth_img: sg.Image = .{},
    depth_view: sg.View = .{}, // depth-stencil attachment
    sample_view: sg.View = .{}, // sampled in the lighting shader
    /// Comparison sampler for hardware PCF (returns 0/1 per tap, filtered).
    compare_sampler: sg.Sampler = .{},
    /// Plain sampler, for debug visualization of raw depth.
    debug_sampler: sg.Sampler = .{},

    /// Quadtree allocator, replacing the shelf packer.
    ///
    /// The shelf packer rebuilt its layout from scratch every frame, so a tile
    /// could land anywhere depending on what was allocated before it. That is
    /// fine while every tile is transient and fatal once any tile is cached,
    /// because a cached tile is only reusable if it is still in the same texels.
    /// This allocator distinguishes the two lifetimes and leaves persistent
    /// tiles untouched by beginFrame.
    tiles: atlas_alloc.AtlasAllocator,

    /// True once any tile in the atlas holds cached depth, which makes the
    /// whole-atlas clear unusable for the frame. Set by the renderer during
    /// build, cleared by reset.
    has_cached_tiles: bool = false,

    /// Takes an allocator now, for the tile tree. Previously this returned by
    /// value with no allocation at all.
    pub fn init(allocator: std.mem.Allocator, opts: AtlasOptions) !ShadowAtlas {
        const size = opts.size;
        const depth_img = sg.makeImage(.{
            .width = @intCast(size),
            .height = @intCast(size),
            .pixel_format = .DEPTH,
            .sample_count = 1,
            // Both a depth attachment (rendered into) AND sampled (in shading).
            .usage = .{ .depth_stencil_attachment = true },
            .label = "shadow_atlas",
        });
        return .{
            .size = size,
            // 64 texels is the smallest tile worth subdividing to. Below that,
            // a shadow map carries too little detail to be useful and the tree
            // overhead outweighs the space saved.
            .tiles = try atlas_alloc.AtlasAllocator.init(allocator, size, 64),
            .depth_img = depth_img,
            .depth_view = sg.makeView(.{ .depth_stencil_attachment = .{ .image = depth_img } }),
            .sample_view = sg.makeView(.{ .texture = .{ .image = depth_img } }),
            .compare_sampler = sg.makeSampler(.{
                .min_filter = .LINEAR, // LINEAR + compare = hardware PCF
                .mag_filter = .LINEAR,
                .wrap_u = .CLAMP_TO_EDGE,
                .wrap_v = .CLAMP_TO_EDGE,
                .compare = .LESS_EQUAL,
                .label = "shadow_compare_sampler",
            }),
            .debug_sampler = sg.makeSampler(.{
                .min_filter = .NEAREST,
                .mag_filter = .NEAREST,
                .wrap_u = .CLAMP_TO_EDGE,
                .wrap_v = .CLAMP_TO_EDGE,
                .label = "shadow_debug_sampler",
            }),
        };
    }

    pub fn deinit(self: *ShadowAtlas) void {
        self.tiles.deinit();
        if (self.debug_sampler.id != 0) sg.destroySampler(self.debug_sampler);
        if (self.compare_sampler.id != 0) sg.destroySampler(self.compare_sampler);
        if (self.sample_view.id != 0) sg.destroyView(self.sample_view);
        if (self.depth_view.id != 0) sg.destroyView(self.depth_view);
        if (self.depth_img.id != 0) sg.destroyImage(self.depth_img);
    }

    pub fn resize(self: *ShadowAtlas, allocator: std.mem.Allocator, opts: AtlasOptions) !void {
        if (opts.size == self.size) return;
        self.deinit();
        self.* = try ShadowAtlas.init(allocator, opts);
    }

    /// Begin a frame's allocation. Releases every transient tile and leaves
    /// persistent ones in place, which is the difference that makes caching
    /// possible.
    pub fn reset(self: *ShadowAtlas) void {
        self.tiles.beginFrame();
        self.has_cached_tiles = false;
    }

    /// Allocate a square tile of `tile_size` texels. Returns null if the atlas
    /// is full, and the caller then skips that light's shadow rather than
    /// failing.
    ///
    /// Sizes are rounded up to a power of two by the allocator, so a request for
    /// 300 yields a 512 tile. Every shadow resolution in use is already a power
    /// of two, so this is not expected to bite, but it does mean the returned
    /// tile size should be used rather than the requested one.
    pub fn allocate(self: *ShadowAtlas, tile_size: u32) ?Tile {
        return self.allocateWithLifetime(tile_size, .transient);
    }

    /// Allocate a tile that survives until explicitly freed. For cached shadows:
    /// the rectangle stays put across frames, which is what makes its depth
    /// reusable.
    ///
    /// The returned handle must be passed to `freeTile` when the tile is no
    /// longer wanted. Nothing here can tell whether a light still exists, so
    /// that responsibility sits with the caller.
    pub fn allocatePersistent(self: *ShadowAtlas, tile_size: u32) ?Tile {
        return self.allocateWithLifetime(tile_size, .persistent);
    }

    /// Release a tile immediately. Persistent tiles normally use this on cache
    /// eviction; transient tiles may use it to roll back a failed multi-tile
    /// allocation in the current frame.
    pub fn freeTile(self: *ShadowAtlas, handle: atlas_alloc.Handle) void {
        self.tiles.free(handle);
    }

    fn allocateWithLifetime(self: *ShadowAtlas, tile_size: u32, lifetime: atlas_alloc.Lifetime) ?Tile {
        const result = self.tiles.allocate(tile_size, lifetime) orelse return null;
        const inv: f32 = 1.0 / @as(f32, @floatFromInt(self.size));
        const r = result.rect;
        return .{
            .x = r.x,
            .y = r.y,
            .size = r.size,
            .handle = result.handle,
            .rect = .{ .uv = .{
                @as(f32, @floatFromInt(r.x)) * inv,
                @as(f32, @floatFromInt(r.y)) * inv,
                @as(f32, @floatFromInt(r.size)) * inv,
                @as(f32, @floatFromInt(r.size)) * inv,
            } },
        };
    }

    pub const Tile = struct {
        x: u32,
        y: u32,
        size: u32,
        rect: AtlasRect,
        /// For releasing a persistent tile. Meaningless for transient ones,
        /// which are released wholesale by reset.
        handle: atlas_alloc.Handle = atlas_alloc.Handle.invalid,
    };

    /// The depth-only pass that clears the WHOLE atlas once per frame. Individual
    /// casters then render into their tiles with a viewport/scissor, LOADing this
    /// cleared depth (so tiles don't clobber each other).
    ///
    /// Incompatible with caching, which needs tiles to survive between frames.
    /// Use loadPass plus clearTile instead when any tile is cached.
    pub fn clearPass(self: ShadowAtlas) sg.Pass {
        var action = sg.PassAction{};
        action.depth = .{ .load_action = .CLEAR, .clear_value = 1.0 };
        var att = sg.Attachments{};
        att.depth_stencil = self.depth_view;
        return .{ .action = action, .attachments = att };
    }

    /// A pass that preserves atlas depth. The basis for caching: tiles that are
    /// still valid are simply not touched; the renderer draws a depth=1
    /// fullscreen triangle under the dirty tile's viewport/scissor before it
    /// redraws that tile's casters.
    pub fn loadPass(self: ShadowAtlas) sg.Pass {
        return self.casterPass();
    }

    /// Convert a tile to a viewport/scissor rectangle. Kept as a small utility
    /// for callers that need to address an atlas region explicitly.
    pub fn tileViewport(tile: Tile, origin_top_left: bool) struct { x: i32, y: i32, w: i32, h: i32, top_left: bool } {
        return .{
            .x = @intCast(tile.x),
            .y = @intCast(tile.y),
            .w = @intCast(tile.size),
            .h = @intCast(tile.size),
            .top_left = origin_top_left,
        };
    }

    /// A caster's pass: LOAD the already-cleared atlas depth and render into it.
    /// The viewport (set separately) confines writes to the caster's tile.
    pub fn casterPass(self: ShadowAtlas) sg.Pass {
        var action = sg.PassAction{};
        action.depth = .{ .load_action = .LOAD };
        var att = sg.Attachments{};
        att.depth_stencil = self.depth_view;
        return .{ .action = action, .attachments = att };
    }
};
