//------------------------------------------------------------------------------
//  shaders/fxaa_quality.glsl
//
//  FXAA 3.11 "quality" variant. Unlike the console variant (fxaa.glsl), this one
//  SEARCHES along the edge in both directions to find where the edge actually
//  ends, then blends proportionally to the pixel's position within that span.
//  That's what preserves texture detail (less over-blur) and handles long, near-
//  horizontal edges — car rooflines, wires, thin spokes — which the cheap variant
//  smears or misses.
//
//  Also adds a SUBPIXEL term: fine detail smaller than a pixel (wheel spokes at
//  distance) can't be resolved by edge blending, so a low-pass of the neighbours
//  is mixed in, capped by subpix_quality.
//
//  Runs on TONEMAPPED (LDR/perceptual) color, after tonemap, before present.
//
//    fs_params (binding 0): inv_resolution.xy = (1/w, 1/h),
//                           .z = edge_threshold, .w = subpix_quality
//    texture: scene_ldr(0); sampler: smp(0, LINEAR clamp — the linear taps at
//             fractional offsets are load-bearing here, not incidental)
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
layout(binding=0) uniform texture2D scene_ldr;
layout(binding=0) uniform sampler smp;

layout(binding=0) uniform fs_params {
    vec4 inv_resolution; // xy = 1/w,1/h   z = edge threshold   w = subpixel amount
};

in vec2 v_uv;
out vec4 frag_color;

#define EDGE_THRESHOLD_MIN 0.0312 // absolute floor: ignore edges in near-black
#define SEARCH_STEPS       12
#define SEARCH_ACCEL       2.0    // step growth once past the first few taps

float luma(vec3 c) {
    return sqrt(dot(c, vec3(0.299, 0.587, 0.114))); // sqrt = rough perceptual curve
}

float lumaAt(vec2 uv) {
    return luma(textureLod(sampler2D(scene_ldr, smp), uv, 0.0).rgb);
}

void main() {
    vec2 inv = inv_resolution.xy;
    float edge_threshold = inv_resolution.z;
    float subpix_quality = inv_resolution.w;

    vec3 rgbM = texture(sampler2D(scene_ldr, smp), v_uv).rgb;
    float lumaM = luma(rgbM);

    // Cross neighbourhood.
    float lumaN = lumaAt(v_uv + vec2(0.0, -inv.y));
    float lumaS = lumaAt(v_uv + vec2(0.0,  inv.y));
    float lumaE = lumaAt(v_uv + vec2( inv.x, 0.0));
    float lumaW = lumaAt(v_uv + vec2(-inv.x, 0.0));

    float lumaMin = min(lumaM, min(min(lumaN, lumaS), min(lumaE, lumaW)));
    float lumaMax = max(lumaM, max(max(lumaN, lumaS), max(lumaE, lumaW)));
    float range = lumaMax - lumaMin;

    // Flat region (or too dark to matter) -> untouched. This early-out is most of
    // the frame, which is why FXAA is cheap.
    if (range < max(EDGE_THRESHOLD_MIN, lumaMax * edge_threshold)) {
        frag_color = vec4(rgbM, 1.0);
        return;
    }

    // Corners, needed for orientation + the subpixel term.
    float lumaNW = lumaAt(v_uv + vec2(-inv.x, -inv.y));
    float lumaNE = lumaAt(v_uv + vec2( inv.x, -inv.y));
    float lumaSW = lumaAt(v_uv + vec2(-inv.x,  inv.y));
    float lumaSE = lumaAt(v_uv + vec2( inv.x,  inv.y));

    float lumaNS = lumaN + lumaS;
    float lumaWE = lumaW + lumaE;
    float lumaNWNE = lumaNW + lumaNE;
    float lumaSWSE = lumaSW + lumaSE;
    float lumaNWSW = lumaNW + lumaSW;
    float lumaNESE = lumaNE + lumaSE;

    // Is the edge horizontal or vertical? Compare gradients across each axis.
    float edgeH = abs(-2.0 * lumaW + lumaNWSW) + abs(-2.0 * lumaM + lumaNS) * 2.0 + abs(-2.0 * lumaE + lumaNESE);
    float edgeV = abs(-2.0 * lumaN + lumaNWNE) + abs(-2.0 * lumaM + lumaWE) * 2.0 + abs(-2.0 * lumaS + lumaSWSE);
    bool horizontal = edgeH >= edgeV;

    // Pick the steeper side; step perpendicular to the edge.
    float luma1 = horizontal ? lumaN : lumaW;
    float luma2 = horizontal ? lumaS : lumaE;
    float grad1 = luma1 - lumaM;
    float grad2 = luma2 - lumaM;
    bool is1 = abs(grad1) >= abs(grad2);
    float gradScaled = 0.25 * max(abs(grad1), abs(grad2));

    float stepLen = horizontal ? inv.y : inv.x;
    float lumaLocalAvg = 0.0;
    if (is1) {
        stepLen = -stepLen;
        lumaLocalAvg = 0.5 * (luma1 + lumaM);
    } else {
        lumaLocalAvg = 0.5 * (luma2 + lumaM);
    }

    // Move to the edge's midline, then search along it.
    vec2 currentUv = v_uv;
    if (horizontal) currentUv.y += stepLen * 0.5;
    else currentUv.x += stepLen * 0.5;

    vec2 offset = horizontal ? vec2(inv.x, 0.0) : vec2(0.0, inv.y);
    vec2 uv1 = currentUv - offset;
    vec2 uv2 = currentUv + offset;

    float lumaEnd1 = lumaAt(uv1) - lumaLocalAvg;
    float lumaEnd2 = lumaAt(uv2) - lumaLocalAvg;
    bool reached1 = abs(lumaEnd1) >= gradScaled;
    bool reached2 = abs(lumaEnd2) >= gradScaled;
    bool reachedBoth = reached1 && reached2;

    if (!reached1) uv1 -= offset;
    if (!reached2) uv2 += offset;

    // Walk outward until both ends of the edge are found (or we run out of steps).
    if (!reachedBoth) {
        for (int i = 2; i < SEARCH_STEPS; i++) {
            if (!reached1) {
                lumaEnd1 = lumaAt(uv1) - lumaLocalAvg;
                reached1 = abs(lumaEnd1) >= gradScaled;
            }
            if (!reached2) {
                lumaEnd2 = lumaAt(uv2) - lumaLocalAvg;
                reached2 = abs(lumaEnd2) >= gradScaled;
            }
            if (reached1 && reached2) break;
            // Accelerate after the first few taps: long edges get found without
            // paying for every texel along them.
            float sc = (i < 4) ? 1.0 : SEARCH_ACCEL;
            if (!reached1) uv1 -= offset * sc;
            if (!reached2) uv2 += offset * sc;
        }
    }

    // Distance to each end; the nearer end drives the blend.
    float dist1 = horizontal ? (v_uv.x - uv1.x) : (v_uv.y - uv1.y);
    float dist2 = horizontal ? (uv2.x - v_uv.x) : (uv2.y - v_uv.y);
    bool dir1 = dist1 < dist2;
    float distFinal = min(dist1, dist2);
    float edgeLen = dist1 + dist2;
    float pixelOffset = -distFinal / max(edgeLen, 1e-6) + 0.5;

    // Only blend if the nearer end's luma sits on the correct side of the edge —
    // otherwise this pixel isn't really on the span we found.
    bool lumaMLess = lumaM < lumaLocalAvg;
    bool correctVariation = ((dir1 ? lumaEnd1 : lumaEnd2) < 0.0) != lumaMLess;
    float finalOffset = correctVariation ? pixelOffset : 0.0;

    // Subpixel term: detail finer than a pixel can't be handled by span blending,
    // so mix in a bounded low-pass of the neighbourhood.
    float lumaAvg = (1.0 / 12.0) * (2.0 * (lumaNS + lumaWE) + lumaNWSW + lumaNESE);
    float subpix1 = clamp(abs(lumaAvg - lumaM) / max(range, 1e-6), 0.0, 1.0);
    float subpix2 = (-2.0 * subpix1 + 3.0) * subpix1 * subpix1; // smoothstep
    float subpixFinal = subpix2 * subpix2 * subpix_quality;
    finalOffset = max(finalOffset, subpixFinal);

    vec2 finalUv = v_uv;
    if (horizontal) finalUv.y += finalOffset * stepLen;
    else finalUv.x += finalOffset * stepLen;

    frag_color = vec4(texture(sampler2D(scene_ldr, smp), finalUv).rgb, 1.0);
}
@end

@program fxaa_quality vs fs
