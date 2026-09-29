// Skinning counterpart of depth_prepass.glsl. Its position arithmetic mirrors
// mesh_skinned.glsl exactly so LESS_EQUAL cannot punch temporal/AO holes.

@include pbr_lib.glsl.inc
@include material_alpha.glsl.inc
@include skinning.glsl.inc

@vs vs
layout(binding=0) uniform vs_params {
    mat4 model;
    mat4 view_proj;
    vec4 uv_scale;
};

in vec3 pos;
in vec3 normal;
in vec2 uv;
in vec4 tangent;
in vec2 uv1;
in uvec4 joints;
in vec4 weights;

out vec3 v_normal;
out vec3 v_tangent;
out float v_tangent_w;
out vec2 v_uv;
out vec2 v_uv1;

@include_block skinning

void main() {
    mat4 skin = zupraSkinMatrix(joints, weights);
    vec4 world = model * (skin * vec4(pos, 1.0));
    mat3 normal_matrix = transpose(inverse(mat3(model * skin)));
    gl_Position = view_proj * world;
    v_normal = normal_matrix * normal;
    v_tangent = mat3(model) * (mat3(skin) * tangent.xyz);
    v_tangent_w = tangent.w;
    v_uv = uv * uv_scale.xy;
    v_uv1 = uv1 * uv_scale.xy;
}
@end

@fs fs
layout(binding=0) uniform texture2D base_color_map;
layout(binding=1) uniform texture2D normal_map;
layout(binding=0) uniform sampler smp_material;

layout(binding=1) uniform fs_params {
    vec4 base_color;
    vec4 alpha_params;
    vec4 normal_params;
};

layout(binding=2) uniform uv_params {
    vec4 uv_m[5];
    vec4 uv_aux[5];
};

in vec3 v_normal;
in vec3 v_tangent;
in float v_tangent_w;
in vec2 v_uv;
in vec2 v_uv1;
out vec4 frag_color;

@include_block uv_transform
@include_block pbr_normal_map
@include_block material_alpha

void main() {
    if (alpha_params.y > 0.5) {
        vec2 uv_bc = mapUv(UV_BASE_COLOR, v_uv, v_uv1);
        vec4 base_sample = texture(sampler2D(base_color_map, smp_material), uv_bc);
        discardMasked(materialAlpha(base_color, base_sample), alpha_params);
    }
    vec2 uv_n = mapUv(UV_NORMAL, v_uv, v_uv1);
    vec3 N = applyNormalMap(normalize(v_normal), v_tangent, v_tangent_w, uv_n, normal_params.x);
    frag_color = vec4(orientTwoSidedNormal(N), 1.0);
}
@end

@program depth_prepass_skinned vs fs
