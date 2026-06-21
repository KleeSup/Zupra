//------------------------------------------------------------------------------
//  shaders/lambert.glsl
//
//  LAMBERT forward shader. Classic diffuse lighting: N·L over the light array
//  (reusing the shared lightAt so point/spot attenuation matches PBR) + a flat
//  ambient term. No specular, no metallic/roughness, no IBL — the "simple lit"
//  model for stylized / low-poly games that don't want PBR realism.
//
//    vs_params (binding 0): model, view_proj
//    fs_params (binding 1): base_color + LightParams (light.zig); the material
//      vec4 is omitted (lambert ignores metallic/roughness). This matches
//      LambertFs in mesh.zig: base_color + LightParams.
//------------------------------------------------------------------------------

@include pbr_lib.glsl.inc

@vs vs
layout(binding=0) uniform vs_params {
    mat4 model;
    mat4 view_proj;
};

in vec3 pos;
in vec3 normal;
in vec2 uv;

out vec3 v_world_pos;
out vec3 v_world_normal;
out vec2 v_uv;

void main() {
    vec4 world = model * vec4(pos, 1.0);
    gl_Position = view_proj * world;
    v_world_pos = world.xyz;
    v_world_normal = mat3(model) * normal;
    v_uv = uv;
}
@end

@fs fs
#define MAX_LIGHTS 16

layout(binding=1) uniform fs_params {
    vec4 base_color;
    vec4 camera_pos;   // unused by lambert; kept so the block matches LightParams
    vec4 ambient_count;
    vec4 light_pos[MAX_LIGHTS];
    vec4 light_dir[MAX_LIGHTS];
    vec4 light_color[MAX_LIGHTS];
    vec4 light_spot[MAX_LIGHTS];
};

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;

out vec4 frag_color;

// Only the light-resolution helper is shared; no BRDF / IBL needed here.
@include_block pbr_lights

void main() {
    vec3 N = normalize(v_world_normal);
    int count = int(ambient_count.w);

    vec3 diffuse = vec3(0.0);
    for (int i = 0; i < count; i++) {
        vec3 L;
        float att;
        lightAt(i, v_world_pos, L, att);
        float ndl = max(dot(N, L), 0.0);
        diffuse += light_color[i].rgb * light_color[i].w * att * ndl;
    }

    vec3 ambient = ambient_count.rgb;
    vec3 color = (ambient + diffuse) * base_color.rgb;
    frag_color = vec4(color, base_color.a);
}
@end

@program lambert vs fs
