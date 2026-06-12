//------------------------------------------------------------------------------
//  shaders/deferred_lighting.glsl
//
//  Deferred LIGHTING pass: fullscreen triangle samples the four G-buffer targets
//  and runs Cook-Torrance PBR per pixel, over a light ARRAY supporting
//  directional, point, and spot lights.
//
//  G-buffer inputs (texture bindings 0..3): albedo, normal(a=geometry mask),
//  position, material(r metallic, g roughness, b ao).
//  light_params (uniform block 0): matches LightParams (light.zig).
//------------------------------------------------------------------------------

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

layout(binding=0) uniform light_params {   // (mesh.glsl: binding=1, and base_color first)
    vec4 camera_pos;
    vec4 ambient_count;
    vec4 light_pos[MAX_LIGHTS];
    vec4 light_dir[MAX_LIGHTS];
    vec4 light_color[MAX_LIGHTS];
    vec4 light_spot[MAX_LIGHTS];
};

in vec2 v_uv;
out vec4 frag_color;

const float PI = 3.14159265359;

float distributionGGX(vec3 N, vec3 H, float rough) {
    float a = rough * rough;
    float a2 = a * a;
    float ndh = max(dot(N, H), 0.0);
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}
float geometrySchlickGGX(float ndv, float rough) {
    float r = rough + 1.0;
    float k = (r * r) / 8.0;
    return ndv / (ndv * (1.0 - k) + k);
}
float geometrySmith(vec3 N, vec3 V, vec3 L, float rough) {
    return geometrySchlickGGX(max(dot(N, V), 0.0), rough) *
           geometrySchlickGGX(max(dot(N, L), 0.0), rough);
}
vec3 fresnelSchlick(float cosT, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosT, 0.0, 1.0), 5.0);
}

void lightAt(int i, vec3 world_pos, out vec3 L, out float att) {
    L = vec3(0.0);   // add this line
    att = 1.0;
    int type = int(light_pos[i].w);
    if (type == 0) {
        L = normalize(-light_dir[i].xyz);
        return;
    }
    vec3 toLight = light_pos[i].xyz - world_pos;
    float dist = length(toLight);
    L = toLight / max(dist, 0.0001);

    float range = light_dir[i].w;
    float a = 1.0 / (dist * dist + 0.0001);
    float window = clamp(1.0 - pow(clamp(dist / range, 0.0, 1.0), 4.0), 0.0, 1.0);
    att = a * window * window;

    if (type == 2) {
        vec3 axis = normalize(light_dir[i].xyz);
        float cosA = dot(normalize(world_pos - light_pos[i].xyz), axis);
        att *= smoothstep(light_spot[i].y, light_spot[i].x, cosA);
    }
}

void main() {
    vec4 nrm = texture(sampler2D(tex_normal, smp), v_uv);
    if (nrm.a < 0.5) {
        discard; // keep the scene-color clear (= clear_color), matching forward
    }
    vec3 albedo = texture(sampler2D(tex_albedo, smp), v_uv).rgb;

    vec3 world_pos = texture(sampler2D(tex_position, smp), v_uv).xyz;
    vec3 mat = texture(sampler2D(tex_material, smp), v_uv).rgb;

    vec3 N = normalize(nrm.xyz);
    float metallic = mat.r;
    float roughness = mat.g;
    float ao = mat.b;

    vec3 V = normalize(camera_pos.xyz - world_pos);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    int count = int(ambient_count.w);
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < count; i++) {
        vec3 L;
        float att;
        lightAt(i, world_pos, L, att);
        vec3 radiance = light_color[i].rgb * light_color[i].w * att;

        vec3 H = normalize(V + L);
        float ndl = max(dot(N, L), 0.0);

        float D = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 spec = (D * G * F) / (4.0 * max(dot(N, V), 0.0) * ndl + 0.0001);
        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

        Lo += (kD * albedo / PI + spec) * radiance * ndl;
    }

    vec3 ambient = ambient_count.rgb * albedo * ao;
    // Output LINEAR HDR — tonemap + gamma happen later in the present pass, so
    // transparent compositing (and future bloom) work in linear space.
    frag_color = vec4(ambient + Lo, 1.0);
}
@end

@program deferred_lighting vs fs
