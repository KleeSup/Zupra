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

out vec3 v_world_normal;
out vec3 v_world_tangent;
out float v_tangent_w;
out vec2 v_uv;
out vec2 v_uv1;

@include_block skinning

void main() {
    mat4 skin = zupraSkinMatrix(joints, weights);
    vec4 world = model * (skin * vec4(pos, 1.0));
    mat3 normal_matrix = transpose(inverse(mat3(model * skin)));
    gl_Position = view_proj * world;
    v_world_normal = normal_matrix * normal;
    v_world_tangent = mat3(model) * (mat3(skin) * tangent.xyz);
    v_tangent_w = tangent.w;
    v_uv = uv * uv_scale.xy;
    v_uv1 = uv1 * uv_scale.xy;
}
@end

@fs fs
layout(binding=0) uniform texture2D normal_map;
layout(binding=1) uniform texture2D base_color_map;
layout(binding=2) uniform texture2D metallic_roughness_map;
layout(binding=3) uniform texture2D occlusion_map;
layout(binding=4) uniform texture2D emissive_map;
layout(binding=0) uniform sampler smp_material;

layout(binding=1) uniform fs_params {
    vec4 base_color;
    vec4 mat_params;
    vec4 emissive;
    vec4 alpha_params;
};

layout(binding=2) uniform uv_params {
    vec4 uv_m[5];
    vec4 uv_aux[5];
};

in vec3 v_world_normal;
in vec3 v_world_tangent;
in float v_tangent_w;
in vec2 v_uv;
in vec2 v_uv1;
layout(location=0) out vec4 g_albedo;
layout(location=1) out vec4 g_normal;
layout(location=2) out vec4 g_material;
layout(location=3) out vec4 g_emissive;

@include_block uv_transform
@include_block pbr_normal_map
@include_block material_alpha

void main() {
    vec2 uv_bc = mapUv(UV_BASE_COLOR, v_uv, v_uv1);
    vec2 uv_n  = mapUv(UV_NORMAL, v_uv, v_uv1);
    vec2 uv_mr = mapUv(UV_METALLIC_ROUGHNESS, v_uv, v_uv1);
    vec2 uv_ao = mapUv(UV_OCCLUSION, v_uv, v_uv1);
    vec2 uv_em = mapUv(UV_EMISSIVE, v_uv, v_uv1);
    vec4 base_sample = texture(sampler2D(base_color_map, smp_material), uv_bc);
    float alpha = materialAlpha(base_color, base_sample);
    discardMasked(alpha, alpha_params);
    vec3 N = normalize(v_world_normal);
    N = applyNormalMap(N, v_world_tangent, v_tangent_w, uv_n, mat_params.w);
    N = orientTwoSidedNormal(N);
    vec3 albedo = base_color.rgb * base_sample.rgb;
    vec3 mr = texture(sampler2D(metallic_roughness_map, smp_material), uv_mr).rgb;
    float metallic = mat_params.x * mr.b;
    float roughness = mat_params.y * mr.g;
    float occ = texture(sampler2D(occlusion_map, smp_material), uv_ao).r;
    float ao = 1.0 + mat_params.z * (occ - 1.0);
    vec3 em = emissive.rgb * emissive.w * texture(sampler2D(emissive_map, smp_material), uv_em).rgb;
    g_albedo = vec4(albedo, alpha);
    g_normal = vec4(N, 1.0);
    g_material = vec4(metallic, roughness, ao, 1.0);
    g_emissive = vec4(em, 1.0);
}
@end

@program gbuffer_skinned vs fs
