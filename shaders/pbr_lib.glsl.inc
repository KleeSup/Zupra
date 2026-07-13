//------------------------------------------------------------------------------
//  shaders/pbr_lib.glsl.inc
//
//  Shared PBR lighting, included by BOTH the forward (mesh.glsl) and deferred
//  (deferred_lighting.glsl) shaders so the two paths compute IDENTICAL lighting
//  and cannot drift. (Not globbed by the build — it ends in .inc, not .glsl.)
//
//  Pulled in with `@include pbr_lib.glsl.inc`, then blocks pasted with
//  `@include_block`.
//
//  CONTRACT — before `@include_block pbr_lights` / `pbr_shade`, the including
//  fragment shader must `#define MAX_LIGHTS`, declare a uniform block with the
//  LightParams members (camera_pos, ambient_count, light_pos[], light_dir[],
//  light_color[], light_spot[]), AND declare the IBL bindings pbr_shade uses:
//      uniform textureCube irradiance_map;   // diffuse IBL
//      uniform textureCube prefilter_map;    // specular IBL (mipped)
//      uniform texture2D   brdf_lut;         // split-sum BRDF (RG)
//      uniform sampler     smp_cube;         // linear + mipmap, clamp
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
// Roughness-aware Fresnel for ambient IBL (rough surfaces dampen grazing rim).
vec3 fresnelSchlickRoughness(float cosT, vec3 F0, float roughness) {
    vec3 r = max(vec3(1.0 - roughness), F0);
    return F0 + (r - F0) * pow(clamp(1.0 - cosT, 0.0, 1.0), 5.0);
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
// Full Cook-Torrance over the light array + IBL ambient (diffuse irradiance +
// specular prefilter*BRDF). The single source of truth for surface shading.
//
// MAX_REFLECTION_LOD must match (prefilter_mips - 1) in ibl.zig.
vec3 pbrShade(vec3 world_pos, vec3 N, vec3 V, vec3 albedo, float metallic, float roughness, float ao) {
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    float ndv = max(dot(N, V), 0.0);

    // --- Direct lights ---
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

        vec3 spec = (D * G * F) / (4.0 * ndv * ndl + 0.0001);
        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
        Lo += (kD * albedo / PI + spec) * radiance * ndl;
    }

    // --- Ambient IBL ---
    vec3 F_amb = fresnelSchlickRoughness(ndv, F0, roughness);
    vec3 kD = (vec3(1.0) - F_amb) * (1.0 - metallic);

    // diffuse: irradiance along N
    vec3 irradiance = texture(samplerCube(irradiance_map, smp_cube), N).rgb;
    vec3 diffuse_ibl = irradiance * albedo;

    // specular: prefiltered reflection * environment BRDF (split-sum)
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 R = reflect(-V, N);
    vec3 prefiltered = textureLod(samplerCube(prefilter_map, smp_cube), R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(sampler2D(brdf_lut, smp_cube), vec2(ndv, roughness)).rg;
    vec3 specular_ibl = prefiltered * (F_amb * brdf.x + brdf.y);

    vec3 ambient = (kD * diffuse_ibl + specular_ibl) * ao;
    return ambient + Lo;
}
@end

@block pbr_hammersley
// Low-discrepancy sequence for importance sampling (prefilter / future bakes).
// Self-contained: defines PI so it can be used without pulling in pbr_brdf.
#ifndef PI
const float PI = 3.14159265359;
#endif

float radicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}
vec2 hammersley(uint i, uint n) {
    return vec2(float(i) / float(n), radicalInverseVdC(i));
}
@end

@block pbr_normal_map
// Perturb the geometric world normal N by a tangent-space normal map. Needs the
// world tangent Tin, handedness w, uv, normal_scale, and the bindings
// `normal_map` (texture2D) + `smp_material` (sampler) declared in the shader.
// If no normal map is bound the default is a flat (0.5,0.5,1) texel -> returns N.
vec3 applyNormalMap(vec3 N, vec3 Tin, float w, vec2 uv, float normal_scale) {
    vec3 T = normalize(Tin - N * dot(N, Tin)); // Gram-Schmidt orthonormalize
    vec3 B = cross(N, T) * w;
    vec3 nm = texture(sampler2D(normal_map, smp_material), uv).xyz * 2.0 - 1.0;
    nm.xy *= abs(normal_scale);
    nm.y *= sign(normal_scale);   // negative normal_scale = DX (flip green)
    return normalize(mat3(T, B, N) * nm);
}
@end
