//------------------------------------------------------------------------------
//  shaders/deferred_lighting.glsl
//
//  Deferred LIGHTING pass: fullscreen triangle samples the four G-buffer targets
//  and runs the SHARED PBR shading (pbr_lib.glsl.inc) over the light array.
//
//  G-buffer inputs (texture bindings 0..3): albedo, normal(a=geometry mask),
//  position, material(r metallic, g roughness, b ao).
//  light_params (uniform block 0): matches LightParams (light.zig).
//------------------------------------------------------------------------------

@include "pbr_lib.glsl.inc"

@vs vs
in vec2 pos;
in vec2 uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
    v_uv = uv;
}
@end

@fs fs
#define MAX_LIGHTS 16

layout(binding=0) uniform texture2D tex_albedo;
layout(binding=1) uniform texture2D tex_normal;
layout(binding=2) uniform texture2D tex_position;
layout(binding=3) uniform texture2D tex_material;
layout(binding=0) uniform sampler smp;

layout(binding=0) uniform light_params {
    vec4 camera_pos;
    vec4 ambient_count;
    vec4 light_pos[MAX_LIGHTS];
    vec4 light_dir[MAX_LIGHTS];
    vec4 light_color[MAX_LIGHTS];
    vec4 light_spot[MAX_LIGHTS];
};

in vec2 v_uv;
out vec4 frag_color;

@include_block pbr_brdf
@include_block pbr_lights
@include_block pbr_shade

void main() {
    vec4 nrm = texture(sampler2D(tex_normal, smp), v_uv);
    if (nrm.a < 0.5) {
        discard; // background: keep the scene-color clear, matching forward
    }

    vec3 albedo = texture(sampler2D(tex_albedo, smp), v_uv).rgb;
    vec3 world_pos = texture(sampler2D(tex_position, smp), v_uv).xyz;
    vec3 mat = texture(sampler2D(tex_material, smp), v_uv).rgb;

    vec3 N = normalize(nrm.xyz);
    vec3 V = normalize(camera_pos.xyz - world_pos);

    // Output LINEAR HDR — tonemap + gamma happen in the present pass.
    vec3 color = pbrShade(world_pos, N, V, albedo, mat.r, mat.g, mat.b);
    frag_color = vec4(color, 1.0);
}
@end

@program deferred_lighting vs fs
