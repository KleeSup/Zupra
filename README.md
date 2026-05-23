# Zupra

The game framework for Zig enthusiasts.

## Dependencies#
- Zig (v0.16.0)
- [bgfx](https://github.com/cyberegoorg/zbgfx)
- [sdl3]()
- [zmath](https://github.com/zig-gamedev/zmath)

## Structure

```
Zupra/
├── build.zig
├── build.zig.zon
│
├── src/
│   ├── root.zig                    # Public API surface (re-exports)
│   │
│   ├── core/
│   │   ├── engine.zig              # Engine: init, main loop, shutdown
│   │   ├── window.zig              # SDL3 window + surface wrapper
│   │   ├── time.zig                # Delta time, frame counter, timers
│   │   ├── event.zig               # SDL3 event pump and framework events
│   │   └── allocator.zig           # Arena/pool allocator setup
│   │
│   ├── input/
│   │   ├── input.zig               # Central input manager
│   │   ├── keyboard.zig            # Key state, just-pressed, just-released
│   │   ├── mouse.zig               # Position, delta, buttons, scroll
│   │   ├── gamepad.zig             # SDL3 gamepad abstraction
│   │   └── action_map.zig          # Bindable action → device agnostic
│   │
│   ├── renderer/
│   │   ├── renderer.zig            # Top-level: init bgfx, submit frame
│   │   ├── backend.zig             # BGFX backend init (SDL3 native handle)
│   │   ├── view.zig                # BGFX view ID management + clear flags
│   │   ├── encoder.zig             # Thread-safe draw call submission wrapper
│   │   │
│   │   ├── resources/
│   │   │   ├── shader.zig          # Load/cache compiled .bin shaders
│   │   │   ├── program.zig         # VS+FS/CS program wrapper
│   │   │   ├── texture.zig         # 2D, cubemap, array texture management
│   │   │   ├── mesh.zig            # Vertex/index buffer pair + layout
│   │   │   └── uniform.zig         # Typed uniform handle cache
│   │   │
│   │   ├── pipeline/
│   │   │   ├── forward.zig         # Forward rendering pass
│   │   │   ├── deferred.zig        # G-buffer fill + lighting resolve pass
│   │   │   ├── framebuffer.zig     # FBO create/resize/blit helpers
│   │   │   └── post_process.zig    # Full-screen effect chain
│   │   │
│   │   ├── batch/
│   │   │   ├── batch2d.zig         # Sprite/quad batcher (texture atlas sort)
│   │   │   ├── batch3d.zig         # Instanced mesh batcher
│   │   │   └── debug_batch.zig     # Lines/rects/circles/spheres (2D+3D)
│   │   │
│   │   ├── effects/
│   │   │   ├── ibl.zig             # Irradiance + prefiltered env map + BRDF LUT
│   │   │   ├── sslr.zig            # Screen-space local reflections
│   │   │   ├── ssao.zig            # Screen-space ambient occlusion
│   │   │   ├── bloom.zig           # Dual-kawase bloom
│   │   │   ├── billboard.zig       # Camera-facing / axis-locked billboards
│   │   │   └── decal.zig           # Deferred decal projection
│   │   │
│   │   ├── material/
│   │   │   ├── material.zig        # PBR material: albedo/metallic/roughness/ao
│   │   │   ├── material_cache.zig  # Handle-based material pool
│   │   │   └── default_materials.zig
│   │   │
│   │   └── camera/
│   │       ├── camera.zig          # View/proj matrix, frustum
│   │       └── camera_controller.zig  # FPS/orbit/cinematic helpers
│   │
│   ├── mesh_builder/
│   │   ├── mesh_builder.zig        # Fluent API: begin/addVertex/addIndex/build
│   │   ├── primitives.zig          # cube, sphere, plane, cylinder, capsule, cone
│   │   └── gltf_loader.zig         # cgltf C binding → internal Mesh
│   │
│   ├── font/
│   │   ├── font_atlas.zig          # SDF/MSDF glyph atlas builder
│   │   ├── text2d.zig              # Screen-space text batching
│   │   └── text3d.zig              # World-space billboard text
│   │
│   └── math/
│       └── math.zig                # zmath re-export + AABB/Ray/Frustum extras
│
├── shaders/
│   ├── src/
│   │   ├── varying.def.sc          # BGFX varying definition shared file
│   │   ├── common/
│   │   │   ├── pbr.sh              # PBR lighting functions (include)
│   │   │   ├── packing.sh          # G-buffer pack/unpack (include)
│   │   │   └── shadows.sh
│   │   ├── 2d/
│   │   │   ├── sprite_vs.sc
│   │   │   └── sprite_fs.sc
│   │   ├── 3d/
│   │   │   ├── mesh_vs.sc
│   │   │   ├── mesh_fs.sc          # Forward PBR
│   │   │   ├── gbuffer_vs.sc
│   │   │   └── gbuffer_fs.sc       # Deferred G-buffer fill
│   │   ├── lighting/
│   │   │   ├── deferred_light_vs.sc
│   │   │   ├── deferred_light_fs.sc
│   │   │   └── ibl_fs.sc
│   │   ├── effects/
│   │   │   ├── sslr_fs.sc
│   │   │   ├── ssao_fs.sc
│   │   │   └── bloom_fs.sc
│   │   ├── font/
│   │   │   ├── sdf_vs.sc
│   │   │   └── sdf_fs.sc
│   │   └── debug/
│   │       ├── line_vs.sc
│   │       └── line_fs.sc
│   └── compiled/                   # build.zig outputs here per-backend
│       ├── glsl/
│       ├── spirv/
│       ├── metal/
│       ├── dx11/
│       └── dx12/
│
├── libs/
│   ├── bgfx/                       # bgfx + bx + bimg source or prebuilt
│   ├── sdl3/                       # SDL3 headers or zig-sdl3 package
│   └── c/
│       ├── stb_truetype.h          # or msdfgen for vector fonts
│       └── cgltf.h                 # glTF loader
│
└── examples/
    ├── hello_2d/
    ├── hello_3d/
    ├── pbr_scene/
    └── deferred_demo/
```