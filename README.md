# Zupra

A rendering and application framework for Zig.

Zupra targets Windows, macOS, Linux, iOS, Android and the web from a single
codebase, on top of sokol's D3D11 / Metal / GL / WebGPU backends. It is a
framework, not an engine: it gives you a renderer, a resource model and an
application loop, and stays out of the way of how you structure a game or tool.

> The 3D renderer is under active development. The lists below mark what is
> implemented, partially implemented, or planned. Percentages are rough
> completeness estimates, not guarantees.

## Overview

Zupra supports both forward and deferred rendering behind one scene interface.
Forward rendering runs through `MeshRenderer` / `ModelBatch`. Deferred rendering
uses the storage / writer / reader split: `GBuffer` (storage), `GeometryRenderer`
(writer) and `DeferredRenderer` (reader). `SceneRenderer` drives either mode and
owns the HDR pipeline, the post-process chain and the light set, so switching
`.forward` and `.deferred` is a one-line change with no other edits.

Lighting is clustered: point and spot lights are assigned to a froxel grid over
the view frustum and each pixel evaluates only the lights that reach it, so there
is no fixed light-count limit. Directional lights and ambient are set on
`Environment`, which is passed to the scene each frame.

Everything renders in linear HDR into an offscreen buffer and is tonemapped at
the end, so bloom, colour grading and other effects operate on real light values.

## Using Zupra as a dependency

Zupra exposes a `zupra` build module. Its BRDF integration LUT is embedded in
the final executable, so a packaged application does not need a `resources`
folder or a particular working directory at runtime.

```zig
const zupra_dep = b.dependency("Zupra", .{
    .target = target,
    .optimize = optimize,
    // Optional: the default is a shipped 512x512 LUT at 4096 samples.
    .brdf_lut_resolution = 256,
    .brdf_lut_samples = 1024,
});
exe.root_module.addImport("zupra", zupra_dep.module("zupra"));
```

The two LUT options are build-time quality settings: non-default values bake a
cache-local asset once and embed it. They are deliberately flat because Zig's
dependency argument syntax does not support nested settings structs.

## Rendering pipeline

### Application & platform
- Windows / macOS / Linux (100%)
- Web (WASM / WebGPU) (100%)
- iOS / Android
- Single-window application loop, input, timing (100%)

### Camera
- Perspective camera (100%)
- Orthographic camera (100%)
- First-person controller (100%)
- Frustum culling (bounding sphere, per submesh) (100%)
- Spatial acceleration for large scenes (BVH / grid)

### Meshes & models
- Static meshes (100%)
- Multi-material models (100%)
- Placed instances sharing one model's GPU buffers (100%)
- GPU instancing: shadow / depth passes (100%)
- GPU instancing: shading passes (needs per-instance material data)
- Mesh builder: cube, plane, sphere (100%)
- Shared-buffer geometry consolidation (one buffer per model, not per primitive)
- Per-mesh bounding volumes (100%)
- Dynamic / streamed meshes
- Skeletal animation & GPU skinning (glTF TRS clips; forward, deferred, depth, shadow and TAA velocity paths) (85%)
- Morph targets
- Level-of-detail (LOD) selection

### Materials
- Metallic-roughness PBR (100%)
- Albedo / normal / metallic-roughness / occlusion / emissive maps (100%)
- Two UV sets + `KHR_texture_transform` (100%)
- Per-texture sampler state (wrap / filter) (100%)
- Alpha blend / mask / opaque modes (100%)
- Per-material custom shaders (forward / self-shaded) (100%)
- Custom shaders that participate in scene lighting (surface-function model)
- `KHR_materials_transmission` (refractive glass)
- Clearcoat / sheen / anisotropy / iridescence

### glTF loading
- Node hierarchy & transforms (100%)
- All standard vertex attributes (100%)
- Full metallic-roughness material set (100%)
- Two UV sets, `KHR_texture_transform`, per-texture samplers (100%)
- Cameras & `KHR_lights_punctual` (100%)
- Embedded & external buffers / textures (100%)
- Sparse accessors (100%)
- Skinning & TRS animation channels (STEP / LINEAR / CUBICSPLINE) (85%)
- Morph targets
- Draco / meshopt / KTX2 (basisu) compression
- Vertex colours (`COLOR_0`), 3+ UV sets

Skinned models keep immutable mesh/clip data on `Model` and use one mutable
`SkeletalAnimator` per independently animated instance:

```zig
var model = try zupra.render.gltf.loadFile(allocator, "character.glb", "assets");
defer model.deinit();

var animator = try model.createAnimator(allocator);
defer animator.deinit();
animator.play(0, true);

var character = model.instance();
character.setAnimator(&animator);

// Each frame, before submitting the instance:
animator.update(delta_seconds);
scene.draw(character);
```

`SceneRenderer` automatically carries the current/previous pose through the
depth, shadow, deferred/forward and TAA-velocity passes. Four-influence
`JOINTS_0` / `WEIGHTS_0` assets are supported today; morph targets, secondary
influence sets and rigid node animation remain the next asset-pipeline work.

### Lighting
- Clustered forward / deferred (froxel grid, per-pixel light lists) (100%)
- Directional, point, spot lights (100%)
- Stable light handles, unbounded scene light count (100%)
- Depth prepass (Forward+): shading runs once per visible fragment (100%)
- Area (rect / tube) lights (storage reserved, LTC evaluation pending)
- Importance-based per-cluster light selection (graceful overflow)
- Cluster occupancy stats (max / average lights per froxel)
- GPU cluster assignment (compute) for very high light counts
- Light cookies / IES profiles

### BRDF
- Cook-Torrance specular (100%)
- GGX distribution (100%)
- Fresnel-Schlick (100%)
- Smith geometry (100%)

### Image-based lighting
- Irradiance map (diffuse) (100%)
- Prefiltered environment map (specular) (100%)
- BRDF integration LUT: generation & sampling (100%)
- Procedural skybox (100%)
- Cubemap loader (50%)
- HDR / equirectangular environment loader (100%)
- Reflection probes (blending, box projection)

### Shadows
- Directional shadow maps, PCF (100%)
- Cascaded shadow maps: stable texel-snapped cascades, cascade blending (100%)
- Per-cascade normal-offset bias, slope-scaled depth bias (100%)
- Spot shadow maps (100%)
- Point-light cubemap shadows (six atlas tiles) (100%)
- Shadow atlas with per-light tile allocation (100%)
- Per-view caster frustum culling + instanced depth passes (100%)
- Distance fade
- Cached static shadows (needs a static / dynamic geometry tag)
- Priority-based atlas allocation under contention
- PCSS / EVSM soft shadows

### Anti-aliasing
- FXAA (100%)
- FXAA 3.11 quality variant (100%)
- SSAA (supersampling) (100%)
- MSAA (forward path) (100%)
- SMAA
- TAA (jittered history + camera/object velocity) (100%)

### Post-processing
- Swappable AA stage (100%)
- User post-effect chain with HDR / LDR injection points (100%)
- Tonemapping (100%)
- Colour grading, chromatic aberration, vignette (examples) (100%)
- Bloom (100%)
- Depth of field
- Motion blur
- Auto-exposure / eye adaptation

### Ambient occlusion
- SSAO (kernel generation, blur)
- XeGTAO (horizon search, denoise) (100%)

### Screen-space reflections
- SSR (ray marching, resolve)
- Reflection-probe fallback

### Global illumination
- Screen-space GI
- Voxel / SDF GI
- Light probes / probe volumes

### 2D
- Sprite batching (100%)
- SDF text rendering (100%)
- Orthographic camera (100%)

### Tooling
- sokol-shdc shader pipeline with shared includes (100%)
- Pipeline-state cache (100%)
- Downstream shader compilation against framework includes (100%)
- Asset hot-reload
- Frame stats overlay: draw counts, culling and instancing ratios (100%)
- Cluster heatmap / froxel occupancy overlay

## Roadmap

Near-term, in order:

1. **Animation maturity:** morph targets, eight-influence (`JOINTS_1` /
   `WEIGHTS_1`) import, rigid animated nodes, and conservative per-joint bounds
   so animated characters can regain aggressive camera/shadow culling.
2. **Material breadth:** glTF clearcoat, transmission, sheen, anisotropy and
   KTX2/BasisU texture paths, with the same linear/HDR colour contract as the
   current PBR route.
3. **Shadow quality/scalability:** priority-aware atlas allocation, higher-end
   soft-shadow options and better cache diagnostics for large retained worlds.

Longer term, one architectural question shapes several of the above. Submission
is immediate-mode: `scene.draw()` records a fresh list each frame and discards it
at `end()`. That keeps the API simple and makes correctness easy to reason about,
but it means culling is a linear scan and a spatial structure would have to be
rebuilt every frame. Scaling past thousands of objects, and caching static
shadow tiles, both point toward a **retained render world** — insert / update /
remove handles for renderables — living alongside the immediate API rather than
replacing it.

Until then, projects with their own broadphase can set
`SceneRenderer.frustum_culling = false` and simply not submit what they have
already rejected. One rule matters when doing so: **do not camera-cull shadow
submissions.** An object behind the camera can still cast a shadow into view.

## Dependencies

- [Zig](https://ziglang.org/) (v0.16.0-dev.3142+5ccfeb926)
- [sokol-zig](https://github.com/floooh/sokol-zig)
- [zstbi](https://github.com/zig-gamedev/zstbi)
- [zmath](https://github.com/zig-gamedev/zmath)
- [zmesh](https://github.com/zig-gamedev/zmesh)
- [stb_truetype.h](https://github.com/nothings/stb/blob/master/stb_truetype.h)
