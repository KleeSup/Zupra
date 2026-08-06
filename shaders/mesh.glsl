//------------------------------------------------------------------------------
//  shaders/mesh.glsl
//
//  Forward PBR mesh shader. Shares pbr_lib.glsl.inc with the deferred path so
//  shading can't diverge. Supports the full glTF material: all five maps,
//  TWO UV SETS (TEXCOORD_0/1) and KHR_texture_transform per map.
//
//  CLUSTERED: punctual lights come from the froxel list, not a uniform array,
//  so this path has no light-count ceiling either. The froxel needs view-space
//  depth, which the vertex shader hands over as gl_Position.w - a standard
//  perspective projection puts view z there, so it's free.
//
//    vs_params (binding 0): model, view_proj, uv_scale
//    fs_params (binding 1): base_color, material(x metallic, y roughness,
//                           z occlusion_strength, w normal_scale), emissive
//    light_params (binding 3): clustered light state (light.zig)
//    uv_params (binding 2): uv_m[5], uv_aux[5] - per-map transform + UV set
//    textures: irradiance(cube 0), prefilter(cube 1), brdf_lut(2), normal(3),
//              base_color(4), emissive(5), metallic_roughness(6), occlusion(7),
//              light_data(8), cluster_table(9), cluster_indices(10)
//    samplers: smp_cube(0), smp_material(1), smp_data(2)
//------------------------------------------------------------------------------

@include pbr_lib.glsl.inc

@vs vs
layout(binding=0) uniform vs_params {
    mat4 model;
    mat4 view_proj;
    vec4 uv_scale; // xy = tiling applied to BOTH uv sets, zw unused
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
    // Perspective divide puts view-space z in w - exactly what the froxel
    // depth slice needs, with no extra uniform or matrix.
    v_view_depth = gl_Position.w;
}
@end

@fs fs
#define MAX_DIRECTIONAL 4
#define ZUPRA_SHADOWS
#define MAX_CASCADES 4
#define MAX_SHADOWED 16

layout(binding=0) uniform textureCube irradiance_map;
layout(binding=1) uniform textureCube prefilter_map;
layout(binding=2) uniform texture2D brdf_lut;
layout(binding=3) uniform texture2D normal_map;
layout(binding=4) uniform texture2D base_color_map;
layout(binding=5) uniform texture2D emissive_map;
layout(binding=6) uniform texture2D metallic_roughness_map;
layout(binding=7) uniform texture2D occlusion_map;
layout(binding=8) uniform texture2D light_data;
layout(binding=9) uniform utexture2D cluster_table;
layout(binding=10) uniform utexture2D cluster_indices;
layout(binding=0) uniform sampler smp_cube;
layout(binding=1) uniform sampler smp_material;
@sampler_type smp_data nonfiltering
layout(binding=2) uniform sampler smp_data;

layout(binding=1) uniform fs_params {
    vec4 base_color;
    vec4 material; // x metallic, y roughness, z occlusion_strength, w normal_scale
    vec4 emissive; // rgb factor, w strength
};

layout(binding=2) uniform uv_params {
    vec4 uv_m[5];   // packed 2x2 (rot*scale) per map
    vec4 uv_aux[5]; // offset.xy, uv_set, unused
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
    vec4 sh_vp[MAX_SHADOWED * MAX_CASCADES];   // mat4 as 4 rows each
    vec4 sh_rect[MAX_SHADOWED * MAX_CASCADES]; // atlas rect per cascade
    vec4 sh_info[MAX_SHADOWED];                // cascades, pcf, blend frac, texel
    vec4 sh_split[MAX_SHADOWED];               // cascade far splits
    vec4 sh_bias[MAX_SHADOWED];                // normal-offset bias per cascade
};

@include_block pbr_brdf
@include_block pbr_lights
@include_block uv_transform
@include_block pbr_normal_map
@include_block pbr_shade

void main() {
    // Resolve each map's UV: pick its set, apply its KHR transform.
    vec2 uv_bc = mapUv(UV_BASE_COLOR, v_uv, v_uv1);
    vec2 uv_n  = mapUv(UV_NORMAL, v_uv, v_uv1);
    vec2 uv_mr = mapUv(UV_METALLIC_ROUGHNESS, v_uv, v_uv1);
    vec2 uv_ao = mapUv(UV_OCCLUSION, v_uv, v_uv1);
    vec2 uv_em = mapUv(UV_EMISSIVE, v_uv, v_uv1);

    vec3 N = normalize(v_world_normal);
    N = applyNormalMap(N, v_world_tangent, v_tangent_w, uv_n, material.w);
    vec3 V = normalize(camera_pos.xyz - v_world_pos);

    vec3 tex = texture(sampler2D(base_color_map, smp_material), uv_bc).rgb;
    vec3 albedo = base_color.rgb * pow(max(tex, vec3(0.0)), vec3(2.2));

    vec3 mr = texture(sampler2D(metallic_roughness_map, smp_material), uv_mr).rgb;
    float metallic = material.x * mr.b;
    float roughness = clamp(material.y * mr.g, 0.04, 1.0);

    float ao_tex = texture(sampler2D(occlusion_map, smp_material), uv_ao).r;
    float ao = mix(1.0, ao_tex, material.z);

    vec3 color = pbrShade(v_world_pos, N, V, albedo, metallic, roughness, ao, v_view_depth);

    vec3 em = emissive.rgb * emissive.w *
              texture(sampler2D(emissive_map, smp_material), uv_em).rgb;
    frag_color = vec4(color + em, base_color.a);
}
@end

@program mesh vs fs
