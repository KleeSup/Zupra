@include pbr_lib.glsl.inc
@include material_alpha.glsl.inc
@include skinning.glsl.inc

@vs vs
layout(binding=0) uniform vs_params {
    mat4 model;
    mat4 view_proj;
    vec4 uv_scale;
};

// `.mesh_skinned_unlit` deliberately uses this compact order: pos, uv, uv1,
// joints, weights. It is not interchangeable with the lit skinned layout.
in vec3 pos;
in vec2 uv;
in vec2 uv1;
in uvec4 joints;
in vec4 weights;

out vec2 v_uv;
out vec2 v_uv1;

@include_block skinning

void main() {
    mat4 skin = zupraSkinMatrix(joints, weights);
    gl_Position = view_proj * (model * (skin * vec4(pos, 1.0)));
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
    vec4 emissive;
    vec4 alpha_params;
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
    vec3 albedo = base_color.rgb * base_sample.rgb;
    vec3 em = emissive.rgb * emissive.w * texture(sampler2D(emissive_map, smp_material), uv_em).rgb;
    frag_color = vec4(albedo + em, alpha);
}
@end

@program unlit_skinned vs fs
