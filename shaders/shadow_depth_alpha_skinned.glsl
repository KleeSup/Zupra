@include pbr_lib.glsl.inc
@include material_alpha.glsl.inc
@include skinning.glsl.inc

@vs vs
layout(binding=0) uniform vs_params {
    mat4 mvp;
    vec4 uv_scale;
};

in vec3 pos;
in vec3 normal;
in vec2 uv;
in vec4 tangent;
in vec2 uv1;
in uvec4 joints;
in vec4 weights;

out vec2 v_uv;
out vec2 v_uv1;

@include_block skinning

void main() {
    mat4 skin = zupraSkinMatrix(joints, weights);
    gl_Position = mvp * (skin * vec4(pos, 1.0));
    v_uv = uv * uv_scale.xy;
    v_uv1 = uv1 * uv_scale.xy;
    gl_Position.x += (normal.x + tangent.x) * 0.0;
}
@end

@fs fs
layout(binding=0) uniform texture2D base_color_map;
layout(binding=0) uniform sampler smp_material;

layout(binding=1) uniform fs_params {
    vec4 base_color;
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
    vec4 base_sample = texture(sampler2D(base_color_map, smp_material), uv_bc);
    discardMasked(materialAlpha(base_color, base_sample), alpha_params);
    frag_color = vec4(1.0);
}
@end

@program shadow_depth_alpha_skinned vs fs
