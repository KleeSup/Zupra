//------------------------------------------------------------------------------
//  shaders/bloom_downsample.glsl
//
//  One step down the bloom mip chain, using the 13-tap filter from Jimenez's
//  Call of Duty presentation.
//
//  WHY 13 TAPS AND NOT A BOX. Halving resolution with a naive 2x2 box aliases
//  badly: features smaller than the new texel alternate between being sampled
//  and skipped as the camera moves, which reads as bloom crawling and pulsing.
//  The 13-tap pattern overlaps a centre group with the four corner groups, so
//  each output texel draws on a wider, smoother neighbourhood and the chain
//  stays stable under motion. It is the standard for exactly that reason.
//
//  The weights sum to 1: this is a pure resample, no energy added or lost.
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
layout(binding=0) uniform downsample_params {
    vec4 params; // xy = 1 / source resolution, zw unused
};

layout(binding=0) uniform texture2D tex_src;
layout(binding=0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

vec3 tap(vec2 uv) {
    return textureLod(sampler2D(tex_src, smp), uv, 0.0).rgb;
}

void main() {
    vec2 t = params.xy;

    // Outer ring, two texels out.
    vec3 a = tap(v_uv + vec2(-2.0, 2.0) * t);
    vec3 b = tap(v_uv + vec2(0.0, 2.0) * t);
    vec3 c = tap(v_uv + vec2(2.0, 2.0) * t);
    vec3 d = tap(v_uv + vec2(-2.0, 0.0) * t);
    vec3 e = tap(v_uv);
    vec3 f = tap(v_uv + vec2(2.0, 0.0) * t);
    vec3 g = tap(v_uv + vec2(-2.0, -2.0) * t);
    vec3 h = tap(v_uv + vec2(0.0, -2.0) * t);
    vec3 i = tap(v_uv + vec2(2.0, -2.0) * t);

    // Inner square, one texel out -- the group that carries most of the weight.
    vec3 j = tap(v_uv + vec2(-1.0, 1.0) * t);
    vec3 k = tap(v_uv + vec2(1.0, 1.0) * t);
    vec3 l = tap(v_uv + vec2(-1.0, -1.0) * t);
    vec3 m = tap(v_uv + vec2(1.0, -1.0) * t);

    // Five overlapping 2x2 boxes: centre at half weight, corners at an eighth.
    vec3 result = (j + k + l + m) * 0.125;
    result += (a + b + d + e) * 0.03125;
    result += (b + c + e + f) * 0.03125;
    result += (d + e + g + h) * 0.03125;
    result += (e + f + h + i) * 0.03125;

    frag_color = vec4(result, 1.0);
}
@end

@program bloom_downsample vs fs
