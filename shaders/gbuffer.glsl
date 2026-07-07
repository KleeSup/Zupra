//------------------------------------------------------------------------------
//  shaders/gbuffer.glsl
//
//  Deferred GEOMETRY pass. Writes surface data to four render targets. Now
//  applies a tangent-space normal map before writing g_normal, so normal-mapped
//  detail is baked into the G-buffer and lit identically by the deferred pass.
//
//  vs_params (binding 0): model, view_proj
//  fs_params (binding 1): base_color, mat_params (x metallic, y roughness,
//                         z ao, w normal_scale)
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
layout(binding=0) uniform texture2D normal_map;
layout(binding=0) uniform sampler smp_material;

layout(binding=1) uniform fs_params {
    vec4 base_color;
    vec4 mat_params; // x metallic, y roughness, z ao, w normal_scale
};

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec3 v_world_tangent;
in float v_tangent_w;
in vec2 v_uv;

layout(location=0) out vec4 g_albedo;
layout(location=1) out vec4 g_normal;
layout(location=2) out vec4 g_position;
layout(location=3) out vec4 g_material;

@include_block pbr_normal_map

void main() {
    vec3 N = normalize(v_world_normal);
    N = applyNormalMap(N, v_world_tangent, v_tangent_w, v_uv, mat_params.w);

    g_albedo = base_color;
    g_normal = vec4(N, 1.0);
    g_position = vec4(v_world_pos, 1.0);
    g_material = vec4(mat_params.xyz, 1.0);
}
@end

@program gbuffer vs fs
