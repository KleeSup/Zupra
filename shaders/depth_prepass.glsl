//------------------------------------------------------------------------------
//  shaders/depth_prepass.glsl
//
//  Depth-only prepass for the forward path. Position in, depth out.
//
//  MUST STAY BIT-IDENTICAL TO mesh.glsl's VERTEX POSITION MATH. The prepass lays
//  down the depth that the shading pass then tests against with LESS_EQUAL, so
//  if the two compute gl_Position differently — even by one ulp — fragments get
//  rejected wherever the prepass rounds nearer, and the object is punched full
//  of holes that reshuffle as the camera moves.
//
//  That rules out the obvious shortcut of premultiplying model * view_proj on
//  the CPU and shipping one mvp matrix: it is the same result in exact
//  arithmetic and a different one in floats. Hence two separate mat4s and the
//  same two multiplies, in the same order, as mesh.glsl.
//
//  Any change to how mesh.glsl derives gl_Position has to be mirrored here.
//
//  Reuses the .mesh vertex layout (pos/normal/uv/tangent/uv1). Only `pos`
//  affects the result, but the other attributes are kept referenced so shdc does
//  NOT strip them — the pipeline binds the full .mesh layout, and a generated
//  vertex-input signature missing attributes would mismatch it and fail
//  pipeline creation.
//------------------------------------------------------------------------------

@vs vs
layout(binding=0) uniform vs_params {
    mat4 model;
    mat4 view_proj;
};

in vec3 pos;
in vec3 normal;
in vec2 uv;
in vec4 tangent;
in vec2 uv1;

void main() {
    vec4 world = model * vec4(pos, 1.0);
    gl_Position = view_proj * world;

    // Keep the unused .mesh attributes live. Applied AFTER the transform, and
    // as an addition of exact zero, so the position arithmetic above stays
    // untouched — perturbing `pos` before the multiply would break the bit
    // equality this pass depends on.
    gl_Position.x += (normal.x + uv.x + tangent.x + uv1.x) * 0.0;
}
@end

@fs fs
out vec4 frag_color;
void main() {
    // Depth-only in practice: the pipeline masks colour writes off, since it
    // shares a pass (and therefore the depth buffer) with the shading draw. An
    // explicit output keeps the fragment stage valid on backends that reject an
    // empty pixel shader.
    frag_color = vec4(1.0);
}
@end

@program depth_prepass vs fs
