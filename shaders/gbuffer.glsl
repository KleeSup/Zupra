//------------------------------------------------------------------------------
//  shaders/gbuffer.glsl
//
//  Deferred GEOMETRY pass. Same vertex stage as mesh.glsl (needs world position
//  + world normal), but the fragment stage writes raw SURFACE DATA to four
//  render targets instead of lighting the pixel. The lighting pass samples
//  these later. Targets match GBuffer's layout:
//    location 0  albedo    (base_color.rgb)
//    location 1  normal    (world-space, RGBA16F so signed values are exact)
//    location 2  position  (world-space)
//    location 3  material  (metallic, roughness, ao)
//
//  vs_params (binding 0): model, view_proj   (same as mesh.glsl)
//  fs_params (binding 1): base_color, mat_params (x metallic, y roughness, z ao)
//------------------------------------------------------------------------------

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
layout(binding=1) uniform fs_params {
    vec4 base_color;
    vec4 mat_params; // x = metallic, y = roughness, z = ao
};

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;

layout(location=0) out vec4 g_albedo;
layout(location=1) out vec4 g_normal;
layout(location=2) out vec4 g_position;
layout(location=3) out vec4 g_material;

void main() {
    g_albedo = base_color;
    g_normal = vec4(normalize(v_world_normal), 1.0);
    g_position = vec4(v_world_pos, 1.0);
    g_material = vec4(mat_params.xyz, 1.0);
}
@end

@program gbuffer vs fs
