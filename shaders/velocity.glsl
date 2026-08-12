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
//------------------------------------------------------------------------------

@vs vs
layout(binding=0) uniform vs_params {
    mat4 model;
    mat4 prev_model;
    mat4 view_proj;
    mat4 prev_view_proj;
};

in vec3 pos;
in vec3 normal;
in vec2 uv;
in vec4 tangent;
in vec2 uv1;

out vec4 v_curr_clip;
out vec4 v_prev_clip;

void main() {
    vec4 world = model * vec4(pos, 1.0);
    vec4 prev_world = prev_model * vec4(pos, 1.0);

    gl_Position = view_proj * world;

    // Passed through as clip positions rather than divided here. The perspective
    // divide is not linear, so interpolating the divided values across the
    // triangle would give the wrong answer everywhere except the vertices.
    v_curr_clip = gl_Position;
    v_prev_clip = prev_view_proj * prev_world;

    // Keep the unused attributes referenced so shdc does not strip them, which
    // would leave a vertex input signature that no longer matches the .mesh
    // layout the pipeline binds. Added after the transform and as an exact zero.
    gl_Position.x += (normal.x + uv.x + tangent.x + uv1.x) * 0.0;
}
@end

@fs fs
// Binding 1, not 0. Uniform block bindings are numbered across the whole
// program rather than per stage, so the vertex stage's vs_params already owns 0.
layout(binding=1) uniform velocity_params {
    // x is 1 when the render target reads top-left first, yzw unused.
    vec4 params;
};

in vec4 v_curr_clip;
in vec4 v_prev_clip;
out vec4 frag_color;

void main() {
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
