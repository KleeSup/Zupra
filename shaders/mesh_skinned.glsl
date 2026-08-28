// Forward PBR vertex skinning variant. Fragment/material bindings intentionally
// match mesh.glsl so only the vertex stream and palette bindings differ.

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

out vec3 v_world_pos;
out vec3 v_world_normal;
out vec3 v_world_tangent;
out float v_tangent_w;
out vec2 v_uv;
out vec2 v_uv1;
out float v_view_depth;

@include_block skinning

void main() {
    mat4 skin = zupraSkinMatrix(joints, weights);
    vec4 world = model * (skin * vec4(pos, 1.0));
    mat3 normal_matrix = transpose(inverse(mat3(model * skin)));
    gl_Position = view_proj * world;
    v_world_pos = world.xyz;
    v_world_normal = normal_matrix * normal;
    v_world_tangent = mat3(model) * (mat3(skin) * tangent.xyz);
    v_tangent_w = tangent.w;
    v_uv = uv * uv_scale.xy;
    v_uv1 = uv1 * uv_scale.xy;
    v_view_depth = gl_Position.w;
}
@end

@fs fs
#define MAX_DIRECTIONAL 4
#define ZUPRA_SHADOWS
#define MAX_CASCADES 4
#define MAX_SHADOW_VIEWS 6
#define MAX_SHADOWED 16

layout(binding=0) uniform textureCube irradiance_map;
layout(binding=1) uniform textureCube prefilter_map;
layout(binding=2) uniform texture2D brdf_lut;
layout(binding=3) uniform texture2D normal_map;
layout(binding=4) uniform texture2D base_color_map;
layout(binding=5) uniform texture2D emissive_map;
layout(binding=6) uniform texture2D metallic_roughness_map;
layout(binding=7) uniform texture2D occlusion_map;
layout(binding=12) uniform texture2D ssao_map;
layout(binding=8) uniform texture2D light_data;
layout(binding=9) uniform utexture2D cluster_table;
layout(binding=10) uniform utexture2D cluster_indices;
layout(binding=0) uniform sampler smp_cube;
layout(binding=1) uniform sampler smp_material;
@sampler_type smp_data nonfiltering
layout(binding=2) uniform sampler smp_data;

layout(binding=1) uniform fs_params {
    vec4 base_color;
    vec4 material;
    vec4 emissive;
    vec4 alpha_params;
};

layout(binding=2) uniform uv_params {
    vec4 uv_m[5];
    vec4 uv_aux[5];
};

layout(binding=3) uniform light_params {
    vec4 camera_pos;
    vec4 ambient_count;
    vec4 light_info;
    vec4 cluster_slice;
    vec4 cluster_grid;
    vec4 cluster_depth;
    vec4 dir_direction[MAX_DIRECTIONAL];
    vec4 dir_color[MAX_DIRECTIONAL];
};

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec3 v_world_tangent;
in float v_tangent_w;
in vec2 v_uv;
in vec2 v_uv1;
in float v_view_depth;
out vec4 frag_color;

@image_sample_type shadow_atlas depth
layout(binding=11) uniform texture2D shadow_atlas;
@sampler_type smp_shadow comparison
layout(binding=3) uniform samplerShadow smp_shadow;

layout(binding=4) uniform shadow_params {
    vec4 sh_vp[MAX_SHADOWED * MAX_SHADOW_VIEWS];
    vec4 sh_rect[MAX_SHADOWED * MAX_SHADOW_VIEWS];
    vec4 sh_info[MAX_SHADOWED];
    vec4 sh_split[MAX_SHADOWED];
    vec4 sh_bias[MAX_SHADOWED];
    vec4 sh_pos[MAX_SHADOWED];
    vec4 sh_fade[MAX_SHADOWED];
};

@include_block pbr_brdf
@include_block pbr_lights
@include_block uv_transform
@include_block pbr_normal_map
@include_block pbr_shade
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
    N = applyNormalMap(N, v_world_tangent, v_tangent_w, uv_n, material.w);
    N = orientTwoSidedNormal(N);
    vec3 V = normalize(camera_pos.xyz - v_world_pos);
    vec3 albedo = base_color.rgb * base_sample.rgb;
    vec3 mr = texture(sampler2D(metallic_roughness_map, smp_material), uv_mr).rgb;
    float metallic = material.x * mr.b;
    float roughness = clamp(material.y * mr.g, 0.04, 1.0);
    float ao_tex = texture(sampler2D(occlusion_map, smp_material), uv_ao).r;
    float ao = mix(1.0, ao_tex, material.z);
    vec2 ao_uv = gl_FragCoord.xy / max(cluster_grid.zw, vec2(1.0));
    ao *= texture(sampler2D(ssao_map, smp_material), ao_uv).r;
    vec3 color = pbrShade(v_world_pos, N, V, albedo, metallic, roughness, ao, v_view_depth);
    vec3 em = emissive.rgb * emissive.w * texture(sampler2D(emissive_map, smp_material), uv_em).rgb;
    frag_color = vec4(color + em, alpha);
}
@end

@program mesh_skinned vs fs
