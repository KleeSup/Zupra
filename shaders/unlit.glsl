//------------------------------------------------------------------------------
//  shaders/unlit.glsl
//
//  UNLIT: albedo + emissive, no lighting. Leanest lit-less material — fetches
//  pos + uv + uv1 (VertexLayout .mesh_unlit; no normal/tangent, since it never
//  lights). Supports TWO UV SETS + KHR_texture_transform on its two maps, so a
//  textured/atlased/tiled material renders the same whether drawn unlit or lit.
//
//    vs_params (binding 0): model, view_proj, uv_scale
//    fs_params (binding 1): base_color, emissive(rgb, w strength)
//    uv_params (binding 2): uv_m[5], uv_aux[5] (uses base_color + emissive)
//    textures: base_color_map(0), emissive_map(1); sampler smp_material(0)
//------------------------------------------------------------------------------

@include pbr_lib.glsl.inc
@include material_alpha.glsl.inc

@vs vs
layout(binding=0) uniform vs_params {
    mat4 model;
    mat4 view_proj;
    vec4 uv_scale;
};

in vec3 pos;
in vec2 uv;
in vec2 uv1;

out vec2 v_uv;
out vec2 v_uv1;

void main() {
    gl_Position = view_proj * (model * vec4(pos, 1.0));
    v_uv = uv * uv_scale.xy;
    v_uv1 = uv1 * uv_scale.xy;
}
@end

@fs fs
layout(binding=0) uniform texture2D base_color_map;
layout(binding=1) uniform texture2D emissive_map;
layout(binding=0) uniform sampler smp_material;

layout(binding=1) uniform fs_params {
    vec4 base_color;
    vec4 emissive; // rgb factor, w strength
    vec4 alpha_params; // x cutoff, y = 1 for glTF alpha MASK
};

layout(binding=2) uniform uv_params {
    vec4 uv_m[5];
    vec4 uv_aux[5];
};

in vec2 v_uv;
in vec2 v_uv1;
out vec4 frag_color;

@include_block uv_transform
@include_block material_alpha

void main() {
    vec2 uv_bc = mapUv(UV_BASE_COLOR, v_uv, v_uv1);
    vec2 uv_em = mapUv(UV_EMISSIVE, v_uv, v_uv1);

    vec4 base_sample = texture(sampler2D(base_color_map, smp_material), uv_bc);
    float alpha = materialAlpha(base_color, base_sample);
    discardMasked(alpha, alpha_params);
    vec3 albedo = base_color.rgb * pow(max(base_sample.rgb, vec3(0.0)), vec3(2.2));
    vec3 em = emissive.rgb * emissive.w * texture(sampler2D(emissive_map, smp_material), uv_em).rgb;
    frag_color = vec4(albedo + em, alpha);
}
@end

@program unlit vs fs
