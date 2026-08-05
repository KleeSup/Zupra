//------------------------------------------------------------------------------
//  shaders/pbr_lib.glsl.inc
//
//  Shared PBR lighting, included by BOTH the forward (mesh.glsl) and deferred
//  (deferred_lighting.glsl) shaders so the two paths compute IDENTICAL lighting
//  and cannot drift. (Not globbed by the build - it ends in .inc, not .glsl.)
//
//  CLUSTERED LIGHTING. Lights are no longer a uniform array. Instead:
//
//    * DIRECTIONAL lights live in the uniform block (they touch every pixel, so
//      clustering them would waste index space on identical data).
//    * PUNCTUAL lights (point/spot) live in a FLOAT TEXTURE, and each pixel
//      looks up only the lights assigned to its froxel.
//
//  Textures rather than storage buffers on purpose: WebGL2/GLES 3.0 have float
//  textures and texelFetch in core but no SSBOs, so this keeps ONE read path on
//  every backend - which is what lets shadows (and everything after) be written
//  once instead of twice.
//
//  CONTRACT - before `@include_block pbr_lights` / `pbr_shade`, the including
//  fragment shader must declare:
//
//    #define MAX_DIRECTIONAL 4     // must match light.zig max_directional
//
//    layout(binding=?) uniform light_params {
//        vec4 camera_pos;
//        vec4 ambient_count;   // rgb ambient, w = directional count
//        vec4 light_info;      // x punctual count, y light tex width (texels),
//                              // z cluster capacity, w unused
//        vec4 cluster_slice;   // x slice scale, y slice bias, z grid.z, w unused
//        vec4 cluster_grid;    // x grid.x, y grid.y, z screen w, w screen h
//        vec4 cluster_depth;   // x near, y far, z origin_top_left flag, w unused
//        vec4 dir_direction[MAX_DIRECTIONAL];
//        vec4 dir_color[MAX_DIRECTIONAL];
//    };
//
//    uniform texture2D  light_data;       // RGBA32F  light store
//    uniform utexture2D cluster_table;    // RG32UI   (offset, count) per froxel
//    uniform utexture2D cluster_indices;  // RGBA32UI 4 light indices per texel
//    uniform sampler    smp_data;         // NEAREST, clamp
//
//  and for pbr_shade, the IBL bindings:
//    uniform textureCube irradiance_map;   // diffuse IBL
//    uniform textureCube prefilter_map;    // specular IBL (mipped)
//    uniform texture2D   brdf_lut;         // split-sum BRDF (RG)
//    uniform sampler     smp_cube;         // linear + mipmap, clamp
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
// ---------------------------------------------------------------------------
//  Light store access. MUST match light.zig's texels_per_light / lights_per_row
//  and cluster.zig's texture layouts - these constants are the CPU<->GPU
//  contract, and a mismatch misreads silently rather than failing loudly.
// ---------------------------------------------------------------------------
#define TEXELS_PER_LIGHT 6
#define LIGHTS_PER_ROW   256
#define CLUSTER_INDEX_TEX_WIDTH 1024

// Fetch texel k (0..5) of punctual light `idx` from the light-data texture.
vec4 lightTexel(int idx, int k) {
    // uint index math: signed integer divide/mod triggers D3D X3556 and is
    // slower, and this runs per-light per-pixel - the hottest path in shading.
    uint uidx = uint(idx);
    uint row = uidx / uint(LIGHTS_PER_ROW);
    uint col = uidx % uint(LIGHTS_PER_ROW);
    return texelFetch(sampler2D(light_data, smp_data),
                      ivec2(int(col) * TEXELS_PER_LIGHT + k, int(row)), 0);
}

// Which froxel a pixel belongs to. Screen tile from gl_FragCoord, depth slice
// from the EXPONENTIAL mapping (linear slices would waste almost every froxel
// on the distant half of the frustum). The scale/bias pair comes from the CPU,
// so both sides derive the same slice by construction.
int froxelIndex(vec2 frag_xy, float view_depth) {
    float gx = cluster_grid.x;
    float gy = cluster_grid.y;
    float gz = cluster_slice.z;

    // gl_FragCoord's y origin differs across backends; the CPU builds tiles in
    // NDC (+y up), so flip when the backend is top-left origin.
    float sy = frag_xy.y;
    if (cluster_depth.z > 0.5) sy = cluster_grid.w - sy;

    int tx = int(clamp(floor(frag_xy.x / cluster_grid.z * gx), 0.0, gx - 1.0));
    int ty = int(clamp(floor(sy / cluster_grid.w * gy), 0.0, gy - 1.0));

    float z = max(view_depth, cluster_depth.x);
    float slice = floor(log(z) * cluster_slice.x + cluster_slice.y);
    int sz = int(clamp(slice, 0.0, gz - 1.0));

    return (sz * int(gy) + ty) * int(gx) + tx;
}

// (offset, count) for a froxel. Index math in uint - signed integer divides
// trigger a D3D "may be slow" warning (X3556) and are genuinely slower.
uvec2 froxelRange(int froxel) {
    uint gx = uint(cluster_grid.x);
    uint f = uint(froxel);
    uint x = f % gx;
    uint y = f / gx;
    return texelFetch(usampler2D(cluster_table, smp_data), ivec2(int(x), int(y)), 0).xy;
}

// Light index at position `i` in the flat index list. Four indices per texel,
// which quarters the fetch count in the inner loop.
int froxelLightIndex(uint i) {
    uint texel = i >> 2u;
    uint comp = i & 3u;
    ivec2 c = ivec2(int(texel % uint(CLUSTER_INDEX_TEX_WIDTH)),
                    int(texel / uint(CLUSTER_INDEX_TEX_WIDTH)));
    uvec4 v = texelFetch(usampler2D(cluster_indices, smp_data), c, 0);
    if (comp == 0u) return int(v.x);
    if (comp == 1u) return int(v.y);
    if (comp == 2u) return int(v.z);
    return int(v.w);
}

// Resolve punctual light `idx`: direction-to-light L and attenuation.
void punctualLightAt(int idx, vec3 world_pos, out vec3 L, out float att) {
    vec4 p = lightTexel(idx, 0); // position.xyz, type
    vec4 d = lightTexel(idx, 1); // direction.xyz, range

    vec3 toLight = p.xyz - world_pos;
    float dist = length(toLight);
    L = toLight / max(dist, 0.0001);

    float range = d.w;
    float a = 1.0 / (dist * dist + 0.0001);
    float window = clamp(1.0 - pow(clamp(dist / range, 0.0, 1.0), 4.0), 0.0, 1.0);
    att = a * window * window;

    if (int(p.w) == 2) { // spot cone
        vec4 s = lightTexel(idx, 3); // cos(inner), cos(outer)
        vec3 axis = normalize(d.xyz);
        float cosA = dot(normalize(world_pos - p.xyz), axis);
        att *= smoothstep(s.y, s.x, cosA);
    }
}

vec3 punctualLightRadiance(int idx) {
    vec4 c = lightTexel(idx, 2); // color.rgb, intensity
    return c.rgb * c.w;
}
@end

@block pbr_shade
// Full Cook-Torrance over the directional lights + this pixel's froxel light
// list, plus IBL ambient (diffuse irradiance + specular prefilter*BRDF). The
// single source of truth for surface shading.
//
// MAX_REFLECTION_LOD must match (prefilter_mips - 1) in ibl.zig.
#define MAX_REFLECTION_LOD 4.0

// One light's Cook-Torrance contribution.
vec3 shadeOneLight(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo,
                   float metallic, float roughness, vec3 F0) {
    float ndl = max(dot(N, L), 0.0);
    if (ndl <= 0.0) return vec3(0.0);

    vec3 H = normalize(V + L);
    float NDF = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 spec = (F * NDF * G) / (4.0 * max(dot(N, V), 0.0) * ndl + 0.0001);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

    return (kD * albedo / PI + spec) * radiance * ndl;
}

// view_depth is the VIEW-SPACE distance along the camera forward axis. In the
// forward path that's gl_Position.w from the vertex shader (a standard
// perspective projection puts view z there); in the deferred path it's the
// linearized G-buffer depth.
vec3 pbrShade(vec3 world_pos, vec3 N, vec3 V, vec3 albedo,
              float metallic, float roughness, float ao, float view_depth) {
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    float ndv = max(dot(N, V), 0.0);
    vec3 Lo = vec3(0.0);

    // --- Directional lights: every pixel, straight from the uniform block ---
    int dir_count = int(ambient_count.w);
    for (int i = 0; i < dir_count; i++) {
        vec3 L = normalize(-dir_direction[i].xyz);
        vec3 radiance = dir_color[i].rgb * dir_color[i].w;
        Lo += shadeOneLight(N, V, L, radiance, albedo, metallic, roughness, F0);
    }

    // --- Punctual lights: only those assigned to this pixel's froxel ---
    int froxel = froxelIndex(gl_FragCoord.xy, view_depth);
    uvec2 range = froxelRange(froxel);
    for (uint i = 0u; i < range.y; i++) {
        int idx = froxelLightIndex(range.x + i);
        vec3 L;
        float att;
        punctualLightAt(idx, world_pos, L, att);
        vec3 radiance = punctualLightRadiance(idx) * att;
        Lo += shadeOneLight(N, V, L, radiance, albedo, metallic, roughness, F0);
    }

    // --- Ambient IBL ---
    vec3 F = fresnelSchlickRoughness(ndv, F0, roughness);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

    vec3 irradiance = texture(samplerCube(irradiance_map, smp_cube), N).rgb;
    vec3 diffuse = irradiance * albedo;

    vec3 R = reflect(-V, N);
    vec3 prefiltered = textureLod(samplerCube(prefilter_map, smp_cube), R,
                                  roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(sampler2D(brdf_lut, smp_cube), vec2(ndv, roughness)).rg;
    vec3 specular = prefiltered * (F * brdf.x + brdf.y);

    vec3 ambient = (kD * diffuse + specular) * ao;
    return ambient + Lo;
}
@end

@block pbr_simple_shade
// Diffuse-only shading + the depth helper. Deliberately SEPARATE from
// pbr_shade: these need only the cluster bindings, not the IBL cubemaps, so a
// shader like lambert.glsl can use them without declaring irradiance_map /
// prefilter_map / brdf_lut that it has no use for.
// Lambert-style diffuse-only accumulation, sharing the same cluster lookup so
// the simple shading model sees exactly the same light set.
vec3 diffuseShade(vec3 world_pos, vec3 N, float view_depth) {
    vec3 diffuse = vec3(0.0);

    int dir_count = int(ambient_count.w);
    for (int i = 0; i < dir_count; i++) {
        vec3 L = normalize(-dir_direction[i].xyz);
        diffuse += dir_color[i].rgb * dir_color[i].w * max(dot(N, L), 0.0);
    }

    int froxel = froxelIndex(gl_FragCoord.xy, view_depth);
    uvec2 range = froxelRange(froxel);
    for (uint i = 0u; i < range.y; i++) {
        int idx = froxelLightIndex(range.x + i);
        vec3 L;
        float att;
        punctualLightAt(idx, world_pos, L, att);
        diffuse += punctualLightRadiance(idx) * att * max(dot(N, L), 0.0);
    }
    return diffuse;
}

// Linearize a [0,1] depth-buffer value to view-space distance. Used by the
// deferred path, which has depth rather than gl_Position.w.
float linearizeDepth(float d) {
    float n = cluster_depth.x;
    float f = cluster_depth.y;
    return n * f / (f - d * (f - n));
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
    // Guard: a degenerate/zero tangent (seams, missing data) would NaN the
    // normal via normalize() and shade the fragment black. Skip if so.
    if (dot(Tin, Tin) < 1e-8) return N;
    vec3 T = normalize(Tin - N * dot(N, Tin)); // Gram-Schmidt orthonormalize
    vec3 B = cross(N, T) * w;
    vec3 nm = texture(sampler2D(normal_map, smp_material), uv).xyz * 2.0 - 1.0;
    nm.xy *= normal_scale;
    return normalize(mat3(T, B, N) * nm);
}
@end

@block uv_transform
// Per-map UV resolution: multiple UV sets (glTF TEXCOORD_0/1) + the
// KHR_texture_transform. The including shader must declare a uv_params block:
//
//     vec4 uv_m[5];    // packed 2x2 (rot*scale): m00, m01, m10, m11
//     vec4 uv_aux[5];  // offset.x, offset.y, uv_set (0|1), unused
//
// Rotation is PRE-BAKED into the 2x2 on the CPU, so there is no sin/cos here —
// just one 2x2 multiply and an add. Indices below match Material.MapIndex.
#define UV_BASE_COLOR         0
#define UV_NORMAL             1
#define UV_METALLIC_ROUGHNESS 2
#define UV_OCCLUSION          3
#define UV_EMISSIVE           4

vec2 mapUv(int i, vec2 uv0, vec2 uv1) {
    vec2 uv = mix(uv0, uv1, uv_aux[i].z); // pick TEXCOORD_0 or TEXCOORD_1
    vec4 m = uv_m[i];
    return vec2(m.x * uv.x + m.y * uv.y,
                m.z * uv.x + m.w * uv.y) + uv_aux[i].xy;
}
@end
