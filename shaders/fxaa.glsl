//------------------------------------------------------------------------------
//  shaders/fxaa.glsl
//
//  FXAA (Fast Approximate Anti-Aliasing) — Timothy Lottes' console variant.
//  A single fullscreen post pass: detect high-contrast edges by luma, then blend
//  along the edge direction. Runs on TONEMAPPED (perceptual/LDR) color — FXAA's
//  luma thresholds are tuned for gamma space, not linear HDR — so in the chain it
//  sits AFTER tonemap, BEFORE present. Path-agnostic: works identically for
//  forward and deferred since it only reads the resolved scene color.
//
//    fs_params (binding 0): inv_resolution.xy = (1/width, 1/height)
//    texture: scene_ldr(0); sampler: smp(0, linear clamp)
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
    vec4 inv_resolution; // xy = 1/width, 1/height
};

in vec2 v_uv;
out vec4 frag_color;

// Tuning constants (Lottes' defaults).
#define FXAA_SPAN_MAX     8.0
#define FXAA_REDUCE_MUL   (1.0 / 8.0)
#define FXAA_REDUCE_MIN   (1.0 / 128.0)
#define FXAA_EDGE_THRESHOLD 0.125  // min luma contrast to treat as an edge

float luma(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

void main() {
    vec2 inv = inv_resolution.xy;

    vec3 rgbNW = texture(sampler2D(scene_ldr, smp), v_uv + vec2(-1.0, -1.0) * inv).rgb;
    vec3 rgbNE = texture(sampler2D(scene_ldr, smp), v_uv + vec2( 1.0, -1.0) * inv).rgb;
    vec3 rgbSW = texture(sampler2D(scene_ldr, smp), v_uv + vec2(-1.0,  1.0) * inv).rgb;
    vec3 rgbSE = texture(sampler2D(scene_ldr, smp), v_uv + vec2( 1.0,  1.0) * inv).rgb;
    vec3 rgbM  = texture(sampler2D(scene_ldr, smp), v_uv).rgb;

    float lumaNW = luma(rgbNW);
    float lumaNE = luma(rgbNE);
    float lumaSW = luma(rgbSW);
    float lumaSE = luma(rgbSE);
    float lumaM  = luma(rgbM);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    // No significant edge here — pass the center pixel through untouched.
    if (lumaMax - lumaMin < lumaMax * FXAA_EDGE_THRESHOLD) {
        frag_color = vec4(rgbM, 1.0);
        return;
    }

    // Edge direction from the luma gradient of the four corners.
    vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * FXAA_REDUCE_MUL, FXAA_REDUCE_MIN);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, vec2(-FXAA_SPAN_MAX), vec2(FXAA_SPAN_MAX)) * inv;

    // Two-tap and four-tap blends along the edge; pick the safe one.
    vec3 rgbA = 0.5 * (
        texture(sampler2D(scene_ldr, smp), v_uv + dir * (1.0 / 3.0 - 0.5)).rgb +
        texture(sampler2D(scene_ldr, smp), v_uv + dir * (2.0 / 3.0 - 0.5)).rgb);
    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture(sampler2D(scene_ldr, smp), v_uv + dir * -0.5).rgb +
        texture(sampler2D(scene_ldr, smp), v_uv + dir *  0.5).rgb);

    float lumaB = luma(rgbB);
    // If the wider blend strayed outside the local luma range, it over-blurred —
    // fall back to the tighter two-tap result.
    if (lumaB < lumaMin || lumaB > lumaMax) {
        frag_color = vec4(rgbA, 1.0);
    } else {
        frag_color = vec4(rgbB, 1.0);
    }
}
@end

@program fxaa vs fs
