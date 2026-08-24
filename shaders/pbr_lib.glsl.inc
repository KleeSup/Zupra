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

// Shadow slot for a punctual light (-1 = unshadowed). Baked into texel 3 .z.
int punctualShadowIndex(int idx) {
    return int(lightTexel(idx, 3).z);
}

// ---------------------------------------------------------------------------
//  Shadows. shadowFactor() returns 1.0 (fully lit) .. 0.0 (fully shadowed).
//  The including shader must, when it wants shadows, declare:
//
//    #define MAX_SHADOW_VIEWS 6
//    #define MAX_SHADOWED 16
//    layout(binding=?) uniform shadow_params {
//        vec4 sh_vp[MAX_SHADOWED * MAX_SHADOW_VIEWS];  // 4 vec4 per mat4
//        vec4 sh_rect[MAX_SHADOWED * MAX_SHADOW_VIEWS];// atlas rect per view
//        vec4 sh_info[MAX_SHADOWED];   // x cascades, y pcf radius, z blend, w texel
//        vec4 sh_split[MAX_SHADOWED];  // cascade far splits (view depth)
//        vec4 sh_bias[MAX_SHADOWED];   // normal-offset bias (world), per cascade
//        vec4 sh_pos[MAX_SHADOWED];    // xyz light position, w = 1 if cube
//        vec4 sh_fade[MAX_SHADOWED];   // x fade start, y 1/span, z local-light bias flag
//    };
//    uniform texture2D shadow_atlas;
//    uniform samplerShadow smp_shadow;  // comparison sampler
//
//  A light's shadow_index (-1 = unshadowed) is carried in the light texture.
//  Directional single-map is cascade 0; CSM picks a cascade by view depth.
// ---------------------------------------------------------------------------
#ifdef ZUPRA_SHADOWS
mat4 shadowMatrix(int shadow_index, int cascade) {
    // sh_vp packs one mat4 as 4 consecutive vec4 ROWS (the CPU side is
    // row-major). mat4() consumes them as COLUMNS, so what comes back is the
    // transpose -- which is exactly what the M * v multiply downstream wants.
    int o = (shadow_index * MAX_SHADOW_VIEWS + cascade) * 4;
    return mat4(sh_vp[o + 0], sh_vp[o + 1], sh_vp[o + 2], sh_vp[o + 3]);
}

// Runtime vec4-component indexing (v[c] with a non-constant c) can't be
// flattened on the glsl300es legacy target, so every per-cascade lookup below
// spells the components out.
//
// Both are written as one initialised local assigned through an if/else chain
// rather than a series of early returns. Equivalent in GLSL, but SPIRV-Cross
// rewrites early returns into a temp guarded by branch flags, and FXC then
// can't prove the temp is assigned on every path -- which produces a stream of
// X4000 "potentially uninitialized variable" warnings on the D3D11 backend.
float cascadeSplit(int shadow_index, int cascade) {
    vec4 splits = sh_split[shadow_index];
    float v = splits.w;
    if (cascade == 0) v = splits.x;
    else if (cascade == 1) v = splits.y;
    else if (cascade == 2) v = splits.z;
    return v;
}

float cascadeNormalBias(int shadow_index, int cascade) {
    vec4 b = sh_bias[shadow_index];
    float v = b.w;
    if (cascade == 0) v = b.x;
    else if (cascade == 1) v = b.y;
    else if (cascade == 2) v = b.z;
    return v;
}

int pickCascade(int shadow_index, float view_depth) {
    // sh_split[shadow_index] packs the up-to-4 cascade far-splits in xyzw.
    // Compare against components explicitly: a runtime vec4-component index
    // (sh_split[..][c]) can't be flattened on the glsl300es legacy target.
    int cascades = int(sh_info[shadow_index].x);
    vec4 splits = sh_split[shadow_index];
    if (cascades > 1 && view_depth < splits.x) return 0;
    if (cascades > 2 && view_depth < splits.y) return 1;
    if (cascades > 3 && view_depth < splits.z) return 2;
    return cascades - 1;
}

float sampleShadow(int shadow_index, int cascade, vec3 world_pos, float ndotl) {
    mat4 lvp = shadowMatrix(shadow_index, cascade);
    // The engine is row-vector (v * M) with row-major storage, and the matrix
    // arrives as four consecutive vec4 ROWS. GLSL's mat4(a,b,c,d) fills COLUMNS,
    // so the reconstruction above already transposes it -- which means `M * v`
    // here performs the row-vector transform. Same form as the model/mvp
    // multiplies in mesh.glsl; writing `v * M` instead silently transposes.
    vec4 lc = lvp * vec4(world_pos, 1.0);
    vec3 proj = lc.xyz / lc.w;

    // Light clip [-1,1] xy -> [0,1], then into the cascade's atlas rect.
    vec2 uv = proj.xy * 0.5 + 0.5;
    if (cluster_depth.z > 0.5) uv.y = 1.0 - uv.y; // match backend depth origin
    vec4 rect = sh_rect[shadow_index * MAX_SHADOW_VIEWS + cascade];

    float depth_ref = proj.z; // sokol [0,1] clip depth

    // Past the far plane there is no data for any light type.
    if (depth_ref > 1.0) return 1.0;

    // Outside the frustum -> lit, but ONLY for light types that have an
    // outside. A spot's cone and a cascade's box both end somewhere, and beyond
    // that edge there is genuinely nothing recorded.
    //
    // A cube has no outside: its six faces tile the sphere completely, so every
    // fragment belongs to exactly one of them. At a face boundary the
    // projection lands exactly on 0 or 1 and rounding pushes it a hair past,
    // and the normal-offset bias nudges the sample point across the seam as
    // well -- so this test would report "lit" along every seam and draw bright
    // lines where the faces meet. The tap clamping below keeps those samples
    // inside their own tile, which is the correct answer there.
    if (sh_pos[shadow_index].w < 0.5) {
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 1.0;
    }

    // PCF: NxN comparison-sampled taps. The comparison sampler returns the
    // fraction of the 2x2 that passes per tap; averaging taps softens further.
    //
    // Taps are offset and CLAMPED in tile space, then mapped into the atlas.
    // Offsetting in atlas space instead lets a tap at a tile border wander into
    // the neighbouring tile, which is a different cascade, a different cube face
    // or a different light entirely -- and reads as a hard wrong-shadow fringe
    // along every tile edge. The half-texel inset accounts for the comparison
    // sampler's own 2x2 footprint, which reaches half a texel past the tap.
    float texel = sh_info[shadow_index].w;
    int radius = int(sh_info[shadow_index].y);
    vec2 tile_texel = vec2(texel) / max(rect.zw, vec2(1e-6));
    vec2 lo = tile_texel * 0.5;
    vec2 hi = vec2(1.0) - lo;

    float sum = 0.0;
    float count = 0.0;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            vec2 tuv = clamp(uv + vec2(float(x), float(y)) * tile_texel, lo, hi);
            vec2 t = rect.xy + tuv * rect.zw;
            sum += textureLod(sampler2DShadow(shadow_atlas, smp_shadow), vec3(t, depth_ref), 0.0);
            count += 1.0;
        }
    }
    return sum / max(count, 1.0);
}

// A perspective shadow-map texel covers proportionally more world space with
// distance from a spot/point light. `normal_bias` is authored at half the
// light range, so local lights scale it to the receiver's actual distance.
// Directional cascades already carry a fitted per-cascade world-space value and
// leave this scale at one.
float shadowReceiverBias(int shadow_index, int cascade, vec3 world_pos) {
    float bias = cascadeNormalBias(shadow_index, cascade);
    if (sh_fade[shadow_index].z > 0.5) {
        float reference_distance = max(cascadeSplit(shadow_index, cascade) * 0.5, 1e-4);
        float receiver_distance = length(world_pos - sh_pos[shadow_index].xyz);
        bias *= clamp(receiver_distance / reference_distance, 0.02, 2.0);
    }
    return bias;
}

vec3 shadowReceiverPosition(int shadow_index, int cascade, vec3 world_pos, vec3 N) {
    return world_pos + N * shadowReceiverBias(shadow_index, cascade, world_pos);
}

float shadowInCascade(int shadow_index, int cascade, vec3 world_pos, vec3 N) {
    vec3 biased = shadowReceiverPosition(shadow_index, cascade, world_pos, N);
    return sampleShadow(shadow_index, cascade, biased, 1.0);
}

// Which cube face a direction falls on, in the order buildPoint fills them:
// +X, -X, +Y, -Y, +Z, -Z. The dominant axis picks the face; its sign picks which
// of the pair. Written as one assigned local rather than early returns, for the
// same FXC reason as cascadeSplit above.
int cubeFace(vec3 dir) {
    vec3 a = abs(dir);
    int face;
    if (a.x >= a.y && a.x >= a.z) {
        face = dir.x > 0.0 ? 0 : 1;
    } else if (a.y >= a.z) {
        face = dir.y > 0.0 ? 2 : 3;
    } else {
        face = dir.z > 0.0 ? 4 : 5;
    }
    return face;
}

// How far the shadow has faded out at this view depth: 0 = full shadow, 1 =
// fully lit. y holds 1/span so this is a multiply, and y = 0 disables the fade
// entirely -- which is how spot and point lights opt out without a branch.
float shadowFade(int shadow_index, float view_depth) {
    vec4 f = sh_fade[shadow_index];
    return clamp((view_depth - f.x) * f.y, 0.0, 1.0);
}

float shadowFactor(int shadow_index, vec3 world_pos, vec3 N, float view_depth) {
    if (shadow_index < 0) return 1.0;

    // Point light: pick the cube face the fragment lies on and sample it. No
    // cascades and nothing to blend -- the six faces tile the sphere edge to
    // edge rather than nesting, so a fragment belongs to exactly one.
    if (sh_pos[shadow_index].w > 0.5) {
        // Select the face from the same biased receiver position that is
        // projected into the atlas. Selecting before the offset can project a
        // sample through one face while clamping it to that face's edge after
        // it has crossed into a neighbour, producing a wrong depth test at a
        // cube seam.
        vec3 biased = shadowReceiverPosition(shadow_index, 0, world_pos, N);
        vec3 to_frag = biased - sh_pos[shadow_index].xyz;
        return sampleShadow(shadow_index, cubeFace(to_frag), biased, 1.0);
    }

    // Nothing left to sample past the end of the range: the outermost cascade
    // has no data out there, so skip the lookup rather than pay for taps whose
    // result the fade would discard anyway.
    float fade = shadowFade(shadow_index, view_depth);
    if (fade >= 1.0) return 1.0;

    int cascades = int(sh_info[shadow_index].x);
    int cascade = pickCascade(shadow_index, view_depth);
    float sh = shadowInCascade(shadow_index, cascade, world_pos, N);

    // CASCADE BLEND. Adjacent cascades differ in resolution, in fitted extent
    // and in bias, so switching between them at a hard threshold draws a visible
    // arc across the ground wherever that threshold falls. Fading across a band
    // just inside the split hides the transition.
    //
    // The band is a fraction of the split distance rather than a fixed width, so
    // it scales with the cascade -- a metre-wide band is generous near the
    // camera and invisibly thin out at the far split.
    //
    // Only fragments inside the band pay for the second lookup; everything else
    // takes the branch once. On the last cascade there's nothing to blend into,
    // and the distance fade (later) takes over instead.
    if (cascade + 1 < cascades) {
        float split = cascadeSplit(shadow_index, cascade);
        float band = split * sh_info[shadow_index].z;
        if (band > 0.0 && view_depth > split - band) {
            float t = clamp((view_depth - (split - band)) / band, 0.0, 1.0);
            float next = shadowInCascade(shadow_index, cascade + 1, world_pos, N);
            sh = mix(sh, next, t);
        }
    }

    // Fade to fully lit at the far end of the range, so the shadow stops being
    // drawn before the point where it would simply stop existing.
    return mix(sh, 1.0, fade);
}
#else
float shadowFactor(int shadow_index, vec3 world_pos, vec3 N, float view_depth) {
    return 1.0; // shadows compiled out
}
#endif

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
        float sh = shadowFactor(int(dir_direction[i].w), world_pos, N, view_depth);
        Lo += shadeOneLight(N, V, L, radiance, albedo, metallic, roughness, F0) * sh;
    }

    // --- Punctual lights: only those assigned to this pixel's froxel ---
    int froxel = froxelIndex(gl_FragCoord.xy, view_depth);
    uvec2 range = froxelRange(froxel);
    for (uint i = 0u; i < range.y; i++) {
        int idx = froxelLightIndex(range.x + i);
        vec3 L;
        float att;
        punctualLightAt(idx, world_pos, L, att);
        // Reject before sampling the shadow map. A froxel holds every light
        // whose volume touches it, so plenty of them contribute nothing to any
        // given fragment -- and each shadowFactor call is a full PCF kernel of
        // dependent atlas fetches, which is the most expensive thing in this
        // loop. Skipping a light that is out of range or behind the surface
        // costs two comparisons and saves nine texture reads.
        if (att <= 1e-4 || dot(N, L) <= 0.0) continue;

        vec3 radiance = punctualLightRadiance(idx) * att;
        float sh = shadowFactor(punctualShadowIndex(idx), world_pos, N, view_depth);
        Lo += shadeOneLight(N, V, L, radiance, albedo, metallic, roughness, F0) * sh;
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
    vec3 result = ambient + Lo;

    return result;
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
        float sh = shadowFactor(int(dir_direction[i].w), world_pos, N, view_depth);
        diffuse += dir_color[i].rgb * dir_color[i].w * max(dot(N, L), 0.0) * sh;
    }

    int froxel = froxelIndex(gl_FragCoord.xy, view_depth);
    uvec2 range = froxelRange(froxel);
    for (uint i = 0u; i < range.y; i++) {
        int idx = froxelLightIndex(range.x + i);
        vec3 L;
        float att;
        punctualLightAt(idx, world_pos, L, att);
        // Same rejection as the PBR loop: no point paying for a shadow lookup
        // for a light that contributes nothing.
        float ndl = dot(N, L);
        if (att <= 1e-4 || ndl <= 0.0) continue;

        float sh = shadowFactor(punctualShadowIndex(idx), world_pos, N, view_depth);
        diffuse += punctualLightRadiance(idx) * att * ndl * sh;
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

    // Second guard, for the case the first one misses. A tangent that is long
    // but parallel to the normal leaves almost nothing after Gram-Schmidt, and
    // normalizing that is NaN just the same. Exported meshes carry this at
    // degenerate UVs, and a single NaN fragment spreads through bloom's mip
    // chain into large black blocks that look nothing like their cause.
    vec3 Tp = Tin - N * dot(N, Tin);
    if (dot(Tp, Tp) < 1e-8) return N;

    vec3 T = normalize(Tp);
    vec3 B = cross(N, T) * w;
    vec3 nm = texture(sampler2D(normal_map, smp_material), uv).xyz * 2.0 - 1.0;
    nm.xy *= normal_scale;

    vec3 mapped = mat3(T, B, N) * nm;
    // Third: a zero or non-finite sample leaves nothing to normalize. Falling
    // back to the geometric normal loses the detail on those fragments, which is
    // the right trade against losing the frame.
    if (!(dot(mapped, mapped) > 1e-12)) return N;
    return normalize(mapped);
}

// glTF double-sided materials draw both windings. The back-facing fragment is
// the reverse side of the same geometric sheet, so it must not retain the
// front-side normal: doing so lets a light on the exterior illuminate/shadow an
// interior wall as though the wall were transparent. Single-sided materials
// cull these fragments before this runs, so the rule is harmless there too.
vec3 orientTwoSidedNormal(vec3 N) {
    return gl_FrontFacing ? N : -N;
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
