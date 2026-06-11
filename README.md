# Zupra

The game framework for Zig enthusiasts.

## 3D Rendering pipeline

Zupra supports both forward as well as deferred rendering. The forward rendering can generally be achieved by using the `MeshRenderer` and the `ModelRenderer`. Deferred rendering requires the **Storage, Writer, Reader-Pattern** which is accomplished by using the `GBuffer` (Storage), `GeometryRenderer` (Writer) and the `DeferredRenderer` (Reader). Lights and ambient color can be set in the `Environment` which is then provided to the selected renderers.

**NOTE: The rendering for 3D is currently under heavily development.**

## Credits to the dependencies used
- [Zig](https://ziglang.org/) (v0.16.0-dev.3142+5ccfeb926)
- [sokol-zig](https://github.com/floooh/sokol-zig)
- [zstbi](https://github.com/zig-gamedev/zstbi)
- [zmath](https://github.com/zig-gamedev/zmath)
- [zmesh](https://github.com/zig-gamedev/zmesh)
- [stb_truetype.h](https://github.com/nothings/stb/blob/master/stb_truetype.h)
