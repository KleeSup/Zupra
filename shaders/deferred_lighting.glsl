//------------------------------------------------------------------------------
//  shaders/deferred_lighting.glsl
//
//  Deferred LIGHTING pass. A fullscreen triangle samples the four G-buffer
//  targets and computes Cook-Torrance PBR per pixel. Directional lights are
//  implemented; the light input is an ARRAY so point/spot are added by
//  extending the loop, not restructuring.
//
//  G-buffer inputs (texture bindings 0..3):
//    tex_albedo   rgb base color
//    tex_normal   xyz world normal (a > 0.5 = geometry present)
//    tex_position xyz world position
//    tex_material r metallic, g roughness, b ao
//
//  light_params (uniform block 0):
//    camera_pos     xyz
//    ambient_count  rgb ambient, w = light count
//    light_dir[i]   xyz direction-to-light (directional), w = type
//    light_color[i] rgb color, w = intensity
//------------------------------------------------------------------------------

@vs vs
in vec2 pos;
in vec2 uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(pos, 0.0, 1.0); // fullscreen triangle, already in clip space
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
    vec4 light_dir[MAX_LIGHTS];
    vec4 light_color[MAX_LIGHTS];
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

void main() {
    vec3 albedo = texture(sampler2D(tex_albedo, smp), v_uv).rgb;
    vec4 nrm = texture(sampler2D(tex_normal, smp), v_uv);

    // Background (no geometry written here): show clear color (black albedo).
    if (nrm.a < 0.5) {
        frag_color = vec4(albedo, 1.0);
        return;
    }

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
        // Directional: light_dir is the direction TO the light.
        vec3 L = normalize(light_dir[i].xyz);
        vec3 radiance = light_color[i].rgb * light_color[i].w;

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
    vec3 color = ambient + Lo;

    // Reinhard tonemap + gamma.
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
    frag_color = vec4(color, 1.0);
}
@end

@program deferred_lighting vs fs
