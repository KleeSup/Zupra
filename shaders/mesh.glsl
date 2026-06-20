//------------------------------------------------------------------------------
//  shaders/mesh.glsl
//
//  FORWARD lit mesh shader. Transform + lighting in one shader (no G-buffer) —
//  the cheap path for simple scenes / mobile / wasm. Uses the SAME shared PBR
//  as the deferred path (pbr_lib.glsl.inc), so forward and deferred match: full
//  Cook-Torrance, metallic/roughness, and point/spot attenuation.
//
//    vs_params (binding 0): model, view_proj  (lighting is world-space)
//    fs_params (binding 1):
//      vec4 base_color                 -- material albedo (per draw)
//      vec4 material                   -- x metallic, y roughness, z ao (per draw)
//      vec4 camera_pos                 -- for specular view vector
//      vec4 ambient_count              -- rgb ambient, w = light count
//      vec4 light_pos[MAX_LIGHTS]      -- xyz pos, w = type
//      vec4 light_dir[MAX_LIGHTS]      -- xyz travel dir, w = range
//      vec4 light_color[MAX_LIGHTS]    -- rgb, w = intensity
//      vec4 light_spot[MAX_LIGHTS]     -- x cos(inner), y cos(outer)
//
//  This block (from base_color onward) matches FsParams in mesh.zig:
//  base_color + material + LightParams (light.zig).
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

layout(binding=0) uniform textureCube irradiance_map;
layout(binding=1) uniform textureCube prefilter_map;
layout(binding=2) uniform texture2D brdf_lut;
layout(binding=0) uniform sampler smp_cube;

layout(binding=1) uniform fs_params {
    vec4 base_color;
    vec4 material; // x metallic, y roughness, z ao
    vec4 camera_pos;
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

@include_block pbr_brdf
@include_block pbr_lights
@include_block pbr_shade

void main() {
    vec3 N = normalize(v_world_normal);
    vec3 V = normalize(camera_pos.xyz - v_world_pos);

    // Output LINEAR HDR — present pass tonemaps.
    vec3 color = pbrShade(v_world_pos, N, V, base_color.rgb, material.x, material.y, material.z);
    frag_color = vec4(color, base_color.a);
}
@end

@program mesh vs fs
