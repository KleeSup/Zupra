//------------------------------------------------------------------------------
//  shaders/pbr_lib.glsl.inc
//
//  Shared PBR lighting, included by BOTH the forward (mesh.glsl) and deferred
//  (deferred_lighting.glsl) shaders so the two paths compute IDENTICAL lighting
//  and cannot drift. (Not globbed by the build — it ends in .inc, not .glsl.)
//
//  Pulled in with `@include pbr_lib.glsl.inc` (no quotes; path is relative to the
//  shaders/ dir), then the blocks are pasted with `@include_block`.
//
//  CONTRACT — before `@include_block pbr_lights` / `pbr_shade`, the including
//  fragment shader must `#define MAX_LIGHTS`, declare a uniform block with the
//  LightParams members (camera_pos, ambient_count, light_pos[], light_dir[],
//  light_color[], light_spot[]), AND declare the IBL bindings used by pbr_shade:
//      uniform textureCube irradiance_map;
//      uniform sampler     smp_cube;
//  Binding slots are the shader's choice; names must match.
//------------------------------------------------------------------------------

@block pbr_brdf
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
@end

@block pbr_lights
// Resolve light i: direction-to-light L and attenuation at world_pos.
void lightAt(int i, vec3 world_pos, out vec3 L, out float att) {
    L = vec3(0.0);
    att = 1.0;
    int type = int(light_pos[i].w);
    if (type == 0) { // directional
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

    if (type == 2) { // spot cone
        vec3 axis = normalize(light_dir[i].xyz);
        float cosA = dot(normalize(world_pos - light_pos[i].xyz), axis);
        att *= smoothstep(light_spot[i].y, light_spot[i].x, cosA);
    }
}
@end

@block pbr_shade
// Full Cook-Torrance over the light array + IBL diffuse ambient. The single
// source of truth for surface shading; both paths resolve the surface, then
// call this.
//
// Ambient is now diffuse IBL: the irradiance map (the environment convolved over
// the hemisphere) sampled along N. Metals get no diffuse (kD = 1 - metallic), so
// they stay dark until SPECULAR IBL (prefilter + BRDF LUT) is added next.
vec3 pbrShade(vec3 world_pos, vec3 N, vec3 V, vec3 albedo, float metallic, float roughness, float ao) {
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

    // Diffuse IBL: environment irradiance along the normal. (1 - metallic) since
    // metals have no diffuse reflection.
    vec3 irradiance = texture(samplerCube(irradiance_map, smp_cube), N).rgb;
    vec3 ambient = (1.0 - metallic) * irradiance * albedo * ao;

    return ambient + Lo;
}
@end
