@include pbr_lib.glsl.inc
@include material_alpha.glsl.inc
@include skinning_velocity.glsl.inc

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
in uvec4 joints;
in vec4 weights;

out vec4 v_curr_clip;
out vec4 v_prev_clip;
out vec2 v_uv;
out vec2 v_uv1;

@include_block skinning_velocity

void main() {
    mat4 current_skin = zupraCurrentSkinMatrix(joints, weights);
    mat4 previous_skin = zupraPreviousSkinMatrix(joints, weights);
    vec4 world = model * (current_skin * vec4(pos, 1.0));
    vec4 prev_world = prev_model * (previous_skin * vec4(pos, 1.0));
    gl_Position = view_proj * world;
    v_curr_clip = gl_Position;
    v_prev_clip = prev_view_proj * prev_world;
    v_uv = uv * uv_scale.xy;
    v_uv1 = uv1 * uv_scale.xy;
    gl_Position.x += (normal.x + tangent.x) * 0.0;
}
@end

@fs fs
layout(binding=0) uniform texture2D base_color_map;
layout(binding=0) uniform sampler smp_material;

layout(binding=1) uniform velocity_params {
    vec4 params;
    vec4 base_color;
    vec4 alpha_params;
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
    frag_color = vec4(prev_uv - curr_uv, 0.0, 1.0);
}
@end

@program velocity_skinned vs fs
