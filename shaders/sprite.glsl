//------------------------------------------------------------------------------
//  shaders/sprite.glsl
//
//  Default 2D sprite shader for the SpriteBatch. Compiled by sokol-shdc into
//  shaders/sprite.glsl.zig (the build globs shaders/*.glsl).
//
//  THIS FILE IS A CONTRACT. The SpriteBatch builds its pipeline against these
//  exact inputs, so any *custom* sprite shader is drop-in compatible as long
//  as it preserves:
//
//    1. Vertex inputs at these locations (must match VertexLayout.sprite in
//       src/graphics/pipeline.zig):
//         location 0  pos    vec2   (Vertex2D.pos,   FLOAT2)
//         location 1  uv     vec2   (Vertex2D.uv,    FLOAT2)
//         location 2  color0 vec4   (Vertex2D.color, UBYTE4N)
//
//    2. The vertex uniform block `vs_params` with a single mat4 `mvp`
//       (camera projection * view; the SpriteBatch uploads this each flush).
//
//    3. One sampled texture `tex` + one `smp` sampler in the fragment stage.
//
//  Change the *bodies* freely (tint math, UV distortion, palette swaps, etc.).
//  Keep the interface above and the batcher won't know the difference.
//------------------------------------------------------------------------------

@vs vs
layout(binding=0) uniform vs_params {
    mat4 mvp;
};

in vec2 pos;
in vec2 uv;
in vec4 color0;

out vec2 v_uv;
out vec4 v_color;

void main() {
    // 2D: z = 0, w = 1. The ortho projection in the camera decides the
    // coordinate convention (pixel space, origin top-left, y-down by default).
    gl_Position = mvp * vec4(pos, 0.0, 1.0);
    v_uv = uv;
    v_color = color0;
}
@end

@fs fs
layout(binding=0) uniform texture2D tex;
layout(binding=0) uniform sampler smp;

in vec2 v_uv;
in vec4 v_color;

out vec4 frag_color;

void main() {
    // Sampled texel modulated by the per-vertex tint.
    frag_color = texture(sampler2D(tex, smp), v_uv) * v_color;
}
@end

@program sprite vs fs
