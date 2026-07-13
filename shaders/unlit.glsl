//------------------------------------------------------------------------------
//  shaders/unlit.glsl
//
//  UNLIT: albedo + emissive, no lighting. The leanest 3D material — fetches only
//  pos + uv (VertexLayout .mesh_unlit; no normal/tangent). For stylized, UI-in-
//  world, billboards, sprites-on-quads, debug. Emits linear HDR (present pass
//  tonemaps), so emissive can drive bloom later.
//
//    vs_params (binding 0): model, view_proj
//    fs_params (binding 1): base_color, emissive(rgb, w strength)
//    textures: base_color_map(0), emissive_map(1); sampler smp_material(0)
//------------------------------------------------------------------------------

@vs vs
layout(binding=0) uniform vs_params {
    mat4 model;
    mat4 view_proj;
    vec4 uv_scale; // xy = texture tiling, zw unused
};

in vec3 pos;
in vec2 uv;

out vec2 v_uv;

void main() {
    gl_Position = view_proj * (model * vec4(pos, 1.0));
    v_uv = uv * uv_scale.xy;
}
@end

@fs fs
layout(binding=0) uniform texture2D base_color_map;
layout(binding=1) uniform texture2D emissive_map;
layout(binding=0) uniform sampler smp_material;

layout(binding=1) uniform fs_params {
    vec4 base_color;
    vec4 emissive; // rgb factor, w = strength
};

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec3 tex = texture(sampler2D(base_color_map, smp_material), v_uv).rgb;
    vec3 albedo = base_color.rgb * pow(max(tex, vec3(0.0)), vec3(2.2)); // sRGB->linear
    vec3 em = emissive.rgb * emissive.w * texture(sampler2D(emissive_map, smp_material), v_uv).rgb;
    frag_color = vec4(albedo + em, base_color.a);
}
@end

@program unlit vs fs
