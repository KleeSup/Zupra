//------------------------------------------------------------------------------
//  shaders/shadow_depth.glsl
//
//  Depth-only shadow caster pass. Position in, depth out — no color target.
//  Fills the shadow atlas with light-space depth as cheaply as possible.
//
//  Reuses the .mesh vertex layout (pos/normal/uv/tangent/uv1). Only `pos` drives
//  the result, but the other attributes are kept referenced (added into a value
//  multiplied by 0) so shdc does NOT strip them — the pipeline binds the full
//  .mesh layout, and a generated vertex-input signature that dropped attributes
//  would mismatch that layout and fail pipeline creation.
//------------------------------------------------------------------------------

@vs vs
layout(binding=0) uniform vs_params {
    mat4 mvp; // model * light_view_proj
};

in vec3 pos;
in vec3 normal;
in vec2 uv;
in vec4 tangent;
in vec2 uv1;

void main() {
    // Keep all .mesh attributes live so the generated input layout matches the
    // pipeline's .mesh vertex layout. The 0.0 factor means they don't affect
    // the position.
    float keep = (normal.x + uv.x + tangent.x + uv1.x) * 0.0;
    gl_Position = mvp * vec4(pos.x + keep, pos.y, pos.z, 1.0);
}
@end

@fs fs
out vec4 frag_color;
void main() {
    // Depth-only in practice (the pipeline uses color_count = 0, so this write
    // is discarded), but an explicit output keeps the fragment stage valid on
    // backends that reject an empty pixel shader.
    frag_color = vec4(1.0);
}
@end

@program shadow_depth vs fs
