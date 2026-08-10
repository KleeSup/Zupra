//------------------------------------------------------------------------------
//  shaders/bloom_upsample.glsl
//
//  One step back up the bloom mip chain: a 3x3 tent filter, blended additively
//  onto the level above.
//
//  WHY PROGRESSIVE UPSAMPLING. A single large Gaussian at one resolution is both
//  expensive and wrong-looking: real lens bloom has no single radius, it has a
//  bright tight core inside a wide faint halo. Summing every mip on the way back
//  up produces exactly that falloff for far less work, because each level
//  contributes its own scale and the widest ones are computed on tiny images.
//
//  The tent radius is in SOURCE texels, so the filter widens in screen terms at
//  each smaller level -- which is what makes the chain's combined response
//  smooth rather than stepped.
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
layout(binding=0) uniform upsample_params {
    vec4 params; // xy = tent radius in uv, z = blend weight, w unused
};

layout(binding=0) uniform texture2D tex_src;
layout(binding=0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

vec3 tap(vec2 uv) {
    return textureLod(sampler2D(tex_src, smp), uv, 0.0).rgb;
}

void main() {
    vec2 r = params.xy;

    // 3x3 tent: 1-2-1 rows, weights summing to 16.
    vec3 result = tap(v_uv + vec2(-1.0, 1.0) * r) * 1.0;
    result += tap(v_uv + vec2(0.0, 1.0) * r) * 2.0;
    result += tap(v_uv + vec2(1.0, 1.0) * r) * 1.0;
    result += tap(v_uv + vec2(-1.0, 0.0) * r) * 2.0;
    result += tap(v_uv) * 4.0;
    result += tap(v_uv + vec2(1.0, 0.0) * r) * 2.0;
    result += tap(v_uv + vec2(-1.0, -1.0) * r) * 1.0;
    result += tap(v_uv + vec2(0.0, -1.0) * r) * 2.0;
    result += tap(v_uv + vec2(1.0, -1.0) * r) * 1.0;
    result *= (1.0 / 16.0);

    // Additive blending is configured on the pipeline, so this pass only has to
    // emit its own contribution -- the level above keeps what it already had.
    frag_color = vec4(result * params.z, 1.0);
}
@end

@program bloom_upsample vs fs
