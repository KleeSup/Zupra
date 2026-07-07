//------------------------------------------------------------------------------
//  shaders/mesh.glsl
//
//  FORWARD lit mesh shader. Uses the shared PBR (pbr_lib.glsl.inc) so it matches
//  the deferred path, and now applies a tangent-space normal map before lighting
//  (same perturbation the G-buffer bakes in the deferred path).
//
//    vs_params (binding 0): model, view_proj
//    fs_params (binding 1): base_color, material(x metallic,y roughness,z ao,
//                           w normal_scale), then LightParams (light.zig)
//    textures: irradiance_map(cube 0), prefilter_map(cube 1), brdf_lut(2d 2),
//              normal_map(2d 3)
//    samplers: smp_cube(0), smp_material(1)
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
in vec4 tangent;

out vec3 v_world_pos;
out vec3 v_world_normal;
out vec3 v_world_tangent;
out float v_tangent_w;
out vec2 v_uv;

void main() {
    vec4 world = model * vec4(pos, 1.0);
    gl_Position = view_proj * world;
    v_world_pos = world.xyz;
    v_world_normal = mat3(model) * normal;
    v_world_tangent = mat3(model) * tangent.xyz;
    v_tangent_w = tangent.w;
    v_uv = uv;
}
@end

@fs fs
#define MAX_LIGHTS 16

layout(binding=0) uniform textureCube irradiance_map;
layout(binding=1) uniform textureCube prefilter_map;
layout(binding=2) uniform texture2D brdf_lut;
layout(binding=3) uniform texture2D normal_map;
layout(binding=0) uniform sampler smp_cube;
layout(binding=1) uniform sampler smp_material;

layout(binding=1) uniform fs_params {
    vec4 base_color;
    vec4 material; // x metallic, y roughness, z ao, w normal_scale
    vec4 camera_pos;
    vec4 ambient_count;
    vec4 light_pos[MAX_LIGHTS];
    vec4 light_dir[MAX_LIGHTS];
    vec4 light_color[MAX_LIGHTS];
    vec4 light_spot[MAX_LIGHTS];
};

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec3 v_world_tangent;
in float v_tangent_w;
in vec2 v_uv;

out vec4 frag_color;

@include_block pbr_brdf
@include_block pbr_lights
@include_block pbr_normal_map
@include_block pbr_shade

void main() {
    vec3 N = normalize(v_world_normal);
    N = applyNormalMap(N, v_world_tangent, v_tangent_w, v_uv, material.w);
    vec3 V = normalize(camera_pos.xyz - v_world_pos);

    vec3 color = pbrShade(v_world_pos, N, V, base_color.rgb, material.x, material.y, material.z);
    frag_color = vec4(color, base_color.a);
}
@end

@program mesh vs fs
