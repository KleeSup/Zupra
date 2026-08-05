//------------------------------------------------------------------------------
//  shaders/deferred_lighting.glsl
//
//  Deferred LIGHTING pass: fullscreen triangle samples the G-buffer and runs the
//  SHARED PBR shading (pbr_lib.glsl.inc), now CLUSTERED, then adds emissive.
//
//  World position is RECONSTRUCTED from the sampled depth + the inverse
//  view-projection (no position target). v_ndc carries clip-space xy from the
//  fullscreen triangle; combined with depth it gives clip space, and
//  inv_view_proj brings it to world.
//
//  The froxel lookup needs VIEW-SPACE depth, which this pass gets by
//  linearizing the sampled G-buffer depth (the forward path instead uses
//  gl_Position.w). Same froxel index either way, so both paths light a given
//  pixel from exactly the same list.
//
//  G-buffer inputs (2D): albedo(0), normal(1, a=mask), material(2), emissive(3),
//                        depth(4). IBL: irradiance(5), prefilter(6), brdf(7).
//  Cluster: light_data(8), cluster_table(9), cluster_indices(10).
//  light_params (ub 0). recon_params (ub 1): inv_view_proj.
//------------------------------------------------------------------------------

@include pbr_lib.glsl.inc

@vs vs
in vec2 pos;
in vec2 uv;
out vec2 v_uv;
out vec2 v_ndc;
void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
    v_uv = uv;
    v_ndc = pos; // fullscreen-tri pos == clip-space xy
}
@end

@fs fs
#define MAX_DIRECTIONAL 4
#define ZUPRA_SHADOWS
#define ZUPRA_SHADOW_DEBUG
#define MAX_CASCADES 4
#define MAX_SHADOWED 16

layout(binding=0) uniform texture2D tex_albedo;
layout(binding=1) uniform texture2D tex_normal;
layout(binding=2) uniform texture2D tex_material;
layout(binding=3) uniform texture2D tex_emissive;
layout(binding=4) uniform texture2D tex_depth;
layout(binding=5) uniform textureCube irradiance_map;
layout(binding=6) uniform textureCube prefilter_map;
layout(binding=7) uniform texture2D brdf_lut;
layout(binding=8) uniform texture2D light_data;
layout(binding=9) uniform utexture2D cluster_table;
layout(binding=10) uniform utexture2D cluster_indices;
layout(binding=0) uniform sampler smp;
layout(binding=1) uniform sampler smp_cube;
@sampler_type smp_data nonfiltering
layout(binding=2) uniform sampler smp_data;

layout(binding=0) uniform light_params {
    vec4 camera_pos;
    vec4 ambient_count;
    vec4 light_info;
    vec4 cluster_slice;
    vec4 cluster_grid;
    vec4 cluster_depth;
    vec4 dir_direction[MAX_DIRECTIONAL];
    vec4 dir_color[MAX_DIRECTIONAL];
};

layout(binding=1) uniform recon_params {
    mat4 inv_view_proj;
};

in vec2 v_uv;
in vec2 v_ndc;
out vec4 frag_color;


@image_sample_type shadow_atlas depth
layout(binding=11) uniform texture2D shadow_atlas;
@sampler_type smp_shadow comparison
layout(binding=3) uniform samplerShadow smp_shadow;

layout(binding=2) uniform shadow_params {
    vec4 sh_vp[MAX_SHADOWED * MAX_CASCADES];   // mat4 as 4 rows each
    vec4 sh_rect[MAX_SHADOWED * MAX_CASCADES]; // atlas rect per cascade
    vec4 sh_info[MAX_SHADOWED];                // cascades, pcf, normal bias, texel
    vec4 sh_split[MAX_SHADOWED];               // cascade far splits
};

@include_block pbr_brdf
@include_block pbr_lights
@include_block pbr_shade
@include_block pbr_simple_shade

void main() {
    vec4 nrm = texture(sampler2D(tex_normal, smp), v_uv);
    if (nrm.a < 0.5) {
        discard; // background: keep the scene-color clear, matching forward
    }

    // Reconstruct world position from depth (sokol [0,1] clip depth, LH).
    float depth = texture(sampler2D(tex_depth, smp), v_uv).r;
    vec4 clip = vec4(v_ndc, depth, 1.0);
    vec4 world_h = inv_view_proj * clip;
    vec3 world_pos = world_h.xyz / world_h.w;

    // View-space depth for the froxel lookup.
    float view_depth = linearizeDepth(depth);

    vec3 albedo = texture(sampler2D(tex_albedo, smp), v_uv).rgb;
    vec3 mat = texture(sampler2D(tex_material, smp), v_uv).rgb;
    vec3 N = normalize(nrm.xyz);
    vec3 V = normalize(camera_pos.xyz - world_pos);

    vec3 color = pbrShade(world_pos, N, V, albedo, mat.r, mat.g, mat.b, view_depth);

    // Emissive (HDR) added after lighting - same linear space, bloom-ready.
    vec3 em = texture(sampler2D(tex_emissive, smp), v_uv).rgb;
    frag_color = vec4(color + em, 1.0);
}
@end

@program deferred_lighting vs fs
