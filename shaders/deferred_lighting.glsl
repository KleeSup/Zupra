//------------------------------------------------------------------------------
//  shaders/deferred_lighting.glsl
//
//  Deferred LIGHTING pass: fullscreen triangle samples the G-buffer and runs the
//  SHARED PBR shading (pbr_lib.glsl.inc) over the light array, then adds emissive.
//
//  World position is RECONSTRUCTED from the sampled depth + the inverse
//  view-projection (no position target). v_ndc carries clip-space xy from the
//  fullscreen triangle; combined with depth it gives clip space, and
//  inv_view_proj brings it to world.
//
//  G-buffer inputs (2D): albedo(0), normal(1, a=mask), material(2), emissive(3),
//                        depth(4). IBL: irradiance(5), prefilter(6), brdf(7).
//  light_params (ub 0): LightParams. recon_params (ub 1): inv_view_proj.
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
#define MAX_LIGHTS 16

layout(binding=0) uniform texture2D tex_albedo;
layout(binding=1) uniform texture2D tex_normal;
layout(binding=2) uniform texture2D tex_material;
layout(binding=3) uniform texture2D tex_emissive;
layout(binding=4) uniform texture2D tex_depth;
layout(binding=5) uniform textureCube irradiance_map;
layout(binding=6) uniform textureCube prefilter_map;
layout(binding=7) uniform texture2D brdf_lut;
layout(binding=0) uniform sampler smp;
layout(binding=1) uniform sampler smp_cube;

layout(binding=0) uniform light_params {
    vec4 camera_pos;
    vec4 ambient_count;
    vec4 light_pos[MAX_LIGHTS];
    vec4 light_dir[MAX_LIGHTS];
    vec4 light_color[MAX_LIGHTS];
    vec4 light_spot[MAX_LIGHTS];
};

layout(binding=1) uniform recon_params {
    mat4 inv_view_proj;
};

in vec2 v_uv;
in vec2 v_ndc;
out vec4 frag_color;

@include_block pbr_brdf
@include_block pbr_lights
@include_block pbr_shade

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

    vec3 albedo = texture(sampler2D(tex_albedo, smp), v_uv).rgb;
    vec3 mat = texture(sampler2D(tex_material, smp), v_uv).rgb;
    vec3 N = normalize(nrm.xyz);
    vec3 V = normalize(camera_pos.xyz - world_pos);

    vec3 color = pbrShade(world_pos, N, V, albedo, mat.r, mat.g, mat.b);

    // Emissive (HDR) added after lighting — same linear space, bloom-ready.
    vec3 em = texture(sampler2D(tex_emissive, smp), v_uv).rgb;
    frag_color = vec4(color + em, 1.0);
}
@end

@program deferred_lighting vs fs
