//------------------------------------------------------------------------------
//  shaders/mesh.glsl
//
//  FORWARD lit mesh shader. Transform AND lighting happen in this one shader
//  (no G-buffer) — the cheap path for simple scenes (few lights). Lighting is
//  Lambert diffuse + ambient over a light ARRAY, so it takes multiple lights;
//  the user picks forward (cheap, few lights) vs deferred (many lights).
//
//  The fragment light block uses the SAME layout as LightParams (light.zig), so
//  one CPU packer (packLightParams) fills it for both forward and deferred.
//  base_color (per-draw material) is prepended; the rest is per-frame light data.
//
//    vs_params (binding 0): model, view_proj   (separate: lighting is world-space)
//    fs_params (binding 1):
//      vec4 base_color                 -- material albedo (per draw)
//      vec4 camera_pos                 -- (unused by Lambert; kept for layout +
//                                         future specular)
//      vec4 ambient_count              -- rgb ambient, w = light count
//      vec4 light_dir[MAX_LIGHTS]      -- xyz direction-TO-light, w = type
//      vec4 light_color[MAX_LIGHTS]    -- rgb, w = intensity
//
//  v_world_pos is emitted now (cheap) so adding point/spot lights later needs
//  no vertex-shader change — only a fragment loop + light_pos[] in LightParams.
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
#define MAX_LIGHTS 16

layout(binding=1) uniform fs_params {
    vec4 base_color;
    vec4 camera_pos;
    vec4 ambient_count;
    vec4 light_dir[MAX_LIGHTS];
    vec4 light_color[MAX_LIGHTS];
};

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;

out vec4 frag_color;

void main() {
    vec3 N = normalize(v_world_normal);

    vec3 lit = base_color.rgb * ambient_count.rgb; // ambient term
    int count = int(ambient_count.w);
    for (int i = 0; i < count; i++) {
        // Directional: light_dir is the direction TO the light.
        vec3 L = normalize(light_dir[i].xyz);
        float ndl = max(dot(N, L), 0.0);
        vec3 radiance = light_color[i].rgb * light_color[i].w; // color * intensity
        lit += base_color.rgb * radiance * ndl;
    }

    frag_color = vec4(lit, base_color.a);
}
@end

@program mesh vs fs
