//------------------------------------------------------------------------------
//  shaders/lambert.glsl
//
//  Diffuse-only forward shading. No specular / metallic / roughness / IBL.
//  Supports two UV sets + KHR_texture_transform on its three maps, so a glTF
//  material renders consistently whichever shading model you pick.
//
//  CLUSTERED: shares the same froxel lookup as the PBR path via
//  diffuseShade(), so a lambert surface and a PBR surface at the same pixel
//  see exactly the same light set - only the BRDF differs.
//
//    vs_params (binding 0): model, view_proj, uv_scale
//    fs_params (binding 1): base_color, material(w = normal_scale), emissive
//    uv_params (binding 2): uv_m[5], uv_aux[5]
//    light_params (binding 3): clustered light state
//    textures: normal(0), base_color(1), emissive(2), light_data(3),
//              cluster_table(4), cluster_indices(5)
//    samplers: smp_material(0), smp_data(1)
//------------------------------------------------------------------------------

@include pbr_lib.glsl.inc

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

out vec3 v_world_pos;
out vec3 v_world_normal;
out vec3 v_world_tangent;
out float v_tangent_w;
out vec2 v_uv;
out vec2 v_uv1;
out float v_view_depth;

void main() {
    vec4 world = model * vec4(pos, 1.0);
    gl_Position = view_proj * world;
    v_world_pos = world.xyz;
    v_world_normal = mat3(model) * normal;
    v_world_tangent = mat3(model) * tangent.xyz;
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
#define MAX_SHADOWED 16

layout(binding=0) uniform texture2D normal_map;
layout(binding=1) uniform texture2D base_color_map;
layout(binding=2) uniform texture2D emissive_map;
layout(binding=3) uniform texture2D light_data;
layout(binding=4) uniform utexture2D cluster_table;
layout(binding=5) uniform utexture2D cluster_indices;
layout(binding=0) uniform sampler smp_material;
@sampler_type smp_data nonfiltering
layout(binding=1) uniform sampler smp_data;

layout(binding=1) uniform fs_params {
    vec4 base_color;
    vec4 material; // w = normal_scale (x/y/z unused by lambert)
    vec4 emissive; // rgb factor, w strength
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


layout(binding=6) uniform texture2D shadow_atlas;
@sampler_type smp_shadow comparison
layout(binding=2) uniform samplerShadow smp_shadow;

layout(binding=4) uniform shadow_params {
    vec4 sh_vp[MAX_SHADOWED * MAX_CASCADES];
    vec4 sh_rect[MAX_SHADOWED * MAX_CASCADES];
    vec4 sh_info[MAX_SHADOWED];
    vec4 sh_split[MAX_SHADOWED];
};

@include_block pbr_brdf
@include_block pbr_lights
@include_block uv_transform
@include_block pbr_normal_map
@include_block pbr_simple_shade

void main() {
    vec2 uv_bc = mapUv(UV_BASE_COLOR, v_uv, v_uv1);
    vec2 uv_n  = mapUv(UV_NORMAL, v_uv, v_uv1);
    vec2 uv_em = mapUv(UV_EMISSIVE, v_uv, v_uv1);

    vec3 N = normalize(v_world_normal);
    N = applyNormalMap(N, v_world_tangent, v_tangent_w, uv_n, material.w);

    vec3 tex = texture(sampler2D(base_color_map, smp_material), uv_bc).rgb;
    vec3 albedo = base_color.rgb * pow(max(tex, vec3(0.0)), vec3(2.2));

    vec3 diffuse = diffuseShade(v_world_pos, N, v_view_depth);

    vec3 color = (ambient_count.rgb + diffuse) * albedo;
    vec3 em = emissive.rgb * emissive.w *
              texture(sampler2D(emissive_map, smp_material), uv_em).rgb;
    frag_color = vec4(color + em, base_color.a);
}
@end

@program lambert vs fs
