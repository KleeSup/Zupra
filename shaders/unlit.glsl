//------------------------------------------------------------------------------
//  shaders/unlit.glsl
//
//  UNLIT forward shader. Flat base color, no lighting at all — for stylized /
//  low-poly / UI-in-world / debug surfaces. No lights, no IBL, no normals used.
//  The cheapest possible 3D material.
//
//    vs_params (binding 0): model, view_proj
//    fs_params (binding 1): base_color
//------------------------------------------------------------------------------

@vs vs
layout(binding=0) uniform vs_params {
    mat4 model;
    mat4 view_proj;
};

in vec3 pos;
in vec3 normal;
in vec2 uv;

out vec2 v_uv;

void main() {
    gl_Position = view_proj * (model * vec4(pos, 1.0));
    v_uv = uv;
}
@end

@fs fs
layout(binding=1) uniform fs_params {
    vec4 base_color;
};

in vec2 v_uv;
out vec4 frag_color;

void main() {
    // Linear HDR out; present pass tonemaps. Unlit = albedo as-is.
    frag_color = base_color;
}
@end

@program unlit vs fs
