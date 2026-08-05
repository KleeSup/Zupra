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
- Frustum culling

### Meshes & models
- Static meshes (100%)
- Multi-material models (100%)
- Instancing (100%)
- Mesh builder: cube, plane, sphere (100%)
- Shared-buffer geometry consolidation (one buffer per model, not per primitive)
- Dynamic / streamed meshes
- Skeletal animation & skinning
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
- Skinning & animation channels
- Morph targets
- Draco / meshopt / KTX2 (basisu) compression
- Vertex colours (`COLOR_0`), 3+ UV sets

### Lighting
- Clustered forward / deferred (froxel grid, per-pixel light lists) (100%)
- Directional, point, spot lights (100%)
- Stable light handles, unbounded scene light count (100%)
- Area (rect / tube) lights (storage reserved, LTC evaluation pending)
- Importance-based per-cluster light selection (graceful overflow)
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
- HDR / equirectangular environment loader
- Reflection probes (blending, box projection)

### Shadows
- Directional shadow map (single cascade), PCF
- Cascaded shadow maps (CSM), stable cascades, cascade blending
- Spot shadow maps
- Point-light cubemap shadows
- Shadow atlas, distance fade, cached static shadows
- PCSS / EVSM soft shadows

### Anti-aliasing
- FXAA (100%)
- FXAA 3.11 quality variant (100%)
- SSAA (supersampling) (100%)
- MSAA (forward path) (100%)
- SMAA
- TAA / TAAU (needs velocity buffer + history)

### Post-processing
- Swappable AA stage (100%)
- User post-effect chain with HDR / LDR injection points (100%)
- Tonemapping (100%)
- Colour grading, chromatic aberration, vignette (examples) (100%)
- Bloom
- Depth of field
- Motion blur
- Auto-exposure / eye adaptation

### Ambient occlusion
- SSAO (kernel generation, blur)
- GTAO (horizon search, denoise)

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
- In-app debug overlays (frame stats, cluster heatmap)

## Roadmap

Near-term, in order:

1. **Shadows:** directional shadow map first (PCF), then cascades, then spot and
   point. The clustered light data is laid out so shadow parameters index in
   parallel with lights, so the shading-side lookup is a small addition.
2. **HDR environment loading:** the IBL bake chain is complete but it needs an
   equirectangular `.hdr` source to replace the procedural sky, which is the
   largest single step up in reflective-material quality.
3. **Bloom:** an HDR-stage post effect (emissive output is already HDR).
4. **SSAO / GTAO:** the deferred G-buffer already carries depth and normals.

Medium-term: skeletal animation and skinning; shared-buffer geometry
consolidation for large scenes; SSR; TAA (via a velocity buffer that also feeds
motion blur and SSR reprojection); reflection probes.

Longer-term: advanced material extensions (transmission, clearcoat, sheen);
Draco / KTX2 asset compression; area lights; global illumination; a broader
tooling and debug-overlay layer.

## Dependencies

- [Zig](https://ziglang.org/) (v0.16.0-dev.3142+5ccfeb926)
- [sokol-zig](https://github.com/floooh/sokol-zig)
- [zstbi](https://github.com/zig-gamedev/zstbi)
- [zmath](https://github.com/zig-gamedev/zmath)
- [zmesh](https://github.com/zig-gamedev/zmesh)
- [stb_truetype.h](https://github.com/nothings/stb/blob/master/stb_truetype.h)