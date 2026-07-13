//------------------------------------------------------------------------------
//  shaders/lambert.glsl
//
//  LAMBERT: albedo + normal map + emissive + diffuse lighting (N·L over the
//  light array, shared lightAt). No specular/metallic/roughness/IBL. The "simple
//  lit" model for stylized/low-poly. Uses the full .mesh layout (needs the
//  tangent for normal mapping).
//
//    vs_params (binding 0): model, view_proj
//    fs_params (binding 1): base_color, material(w = normal_scale), emissive,
//                           then LightParams (light.zig)
//    textures: normal_map(0), base_color_map(1), emissive_map(2)
//    sampler:  smp_material(0)
//------------------------------------------------------------------------------

@include pbr_lib.glsl.inc

@vs vs
layout(binding=0) uniform vs_params {
    mat4 model;
    mat4 view_proj;
    vec4 uv_scale; // xy = texture tiling, zw unused
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
    v_uv = uv * uv_scale.xy;
}
@end

@fs fs
#define MAX_LIGHTS 16

layout(binding=0) uniform texture2D normal_map;
layout(binding=1) uniform texture2D base_color_map;
layout(binding=2) uniform texture2D emissive_map;
layout(binding=0) uniform sampler smp_material;

layout(binding=1) uniform fs_params {
    vec4 base_color;
    vec4 material; // w = normal_scale (x/y/z unused by lambert)
    vec4 emissive; // rgb factor, w strength
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

@include_block pbr_lights
@include_block pbr_normal_map

void main() {
    vec3 N = normalize(v_world_normal);
    N = applyNormalMap(N, v_world_tangent, v_tangent_w, v_uv, material.w);

    vec3 tex = texture(sampler2D(base_color_map, smp_material), v_uv).rgb;
    vec3 albedo = base_color.rgb * pow(max(tex, vec3(0.0)), vec3(2.2));

    int count = int(ambient_count.w);
    vec3 diffuse = vec3(0.0);
    for (int i = 0; i < count; i++) {
        vec3 L;
        float att;
        lightAt(i, v_world_pos, L, att);
        diffuse += light_color[i].rgb * light_color[i].w * att * max(dot(N, L), 0.0);
    }

    vec3 color = (ambient_count.rgb + diffuse) * albedo;
    vec3 em = emissive.rgb * emissive.w * texture(sampler2D(emissive_map, smp_material), v_uv).rgb;
    frag_color = vec4(color + em, base_color.a);
}
@end

@program lambert vs fs
