//------------------------------------------------------------------------------
//  shaders/velocity.glsl
//
//  Per-object motion vectors.
//
//  Writes, for each pixel, how far the surface at that pixel moved on screen
//  since the previous frame, measured in UV units. TAA already reprojects using
//  camera motion, which is exact for anything that stayed still, but an object
//  moving under its own transform has no camera-derived velocity and its history
//  is fetched from the wrong place. The neighbourhood clamp hides most of that,
//  and what it cannot hide shows up as ghosting behind moving geometry.
//
//  The vertex shader transforms the same vertex twice, once by the current model
//  matrix and once by the previous one, and the fragment shader takes the
//  difference of the two screen positions. Doing it per vertex rather than per
//  object is what makes it correct for rotation and scaling as well as
//  translation, since different parts of a rotating object move at different
//  rates.
//
//  Both matrices are unjittered. The jitter is a rendering offset rather than
//  part of where a surface actually is, and leaving it in would add the
//  difference between two frames of the jitter sequence to every velocity.
//
//  Alpha-masked geometry samples and discards with exactly the same glTF
//  base-colour-alpha rule as its colour/depth/shadow passes. Otherwise a moving
//  leaf writes a velocity vector through its cut-out holes and TAA reprojects
//  the background incorrectly.
//------------------------------------------------------------------------------

@include pbr_lib.glsl.inc
@include material_alpha.glsl.inc

@vs vs
layout(binding=0) uniform vs_params {
    mat4 model;
    mat4 prev_model;
    mat4 view_proj;
    mat4 prev_view_proj;
    vec4 uv_scale;
};

in vec3 pos;
in vec3 normal;
in vec2 uv;
in vec4 tangent;
in vec2 uv1;

out vec4 v_curr_clip;
out vec4 v_prev_clip;
out vec2 v_uv;
out vec2 v_uv1;

void main() {
    vec4 world = model * vec4(pos, 1.0);
    vec4 prev_world = prev_model * vec4(pos, 1.0);

    gl_Position = view_proj * world;

    // Passed through as clip positions rather than divided here. The perspective
    // divide is not linear, so interpolating the divided values across the
    // triangle would give the wrong answer everywhere except the vertices.
    v_curr_clip = gl_Position;
    v_prev_clip = prev_view_proj * prev_world;
    v_uv = uv * uv_scale.xy;
    v_uv1 = uv1 * uv_scale.xy;

    // Keep the unused attributes referenced so shdc does not strip them, which
    // would leave a vertex input signature that no longer matches the .mesh
    // layout the pipeline binds. Added after the transform and as an exact zero.
    gl_Position.x += (normal.x + tangent.x) * 0.0;
}
@end

@fs fs
layout(binding=0) uniform texture2D base_color_map;
layout(binding=0) uniform sampler smp_material;

// Binding 1, not 0. Uniform block bindings are numbered across the whole
// program rather than per stage, so the vertex stage's vs_params already owns 0.
layout(binding=1) uniform velocity_params {
    // x is 1 when the render target reads top-left first, yzw unused.
    vec4 params;
    vec4 base_color;
    vec4 alpha_params; // x cutoff, y = 1 for glTF alpha MASK
};

layout(binding=2) uniform uv_params {
    vec4 uv_m[5];
    vec4 uv_aux[5];
};

in vec4 v_curr_clip;
in vec4 v_prev_clip;
in vec2 v_uv;
in vec2 v_uv1;
out vec4 frag_color;

@include_block uv_transform
@include_block material_alpha

void main() {
    // Static opaque objects never reach this pass, but moved opaque objects do;
    // avoid an otherwise unnecessary texture read for them.
    if (alpha_params.y > 0.5) {
        vec2 uv_bc = mapUv(UV_BASE_COLOR, v_uv, v_uv1);
        vec4 base_sample = texture(sampler2D(base_color_map, smp_material), uv_bc);
        discardMasked(materialAlpha(base_color, base_sample), alpha_params);
    }

    vec2 curr_ndc = v_curr_clip.xy / v_curr_clip.w;
    vec2 prev_ndc = v_prev_clip.xy / v_prev_clip.w;

    vec2 curr_uv = curr_ndc * 0.5 + 0.5;
    vec2 prev_uv = prev_ndc * 0.5 + 0.5;
    if (params.x > 0.5) {
        curr_uv.y = 1.0 - curr_uv.y;
        prev_uv.y = 1.0 - prev_uv.y;
    }

    // Stored as the offset from this pixel to where the surface was, so the
    // resolve can add it directly to its own UV. Storing the forward direction
    // instead would make every consumer negate it.
    frag_color = vec4(prev_uv - curr_uv, 0.0, 1.0);
}
@end

@program velocity vs fs
