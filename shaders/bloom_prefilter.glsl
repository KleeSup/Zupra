//------------------------------------------------------------------------------
//  shaders/bloom_prefilter.glsl
//
//  Bright-pass: isolates the part of the HDR image that should glow, and
//  downsamples to half resolution in the same pass.
//
//  SOFT KNEE. A hard `if (luma > threshold)` cut makes bloom pop in and out as a
//  surface crosses the threshold -- a light dimming through it flickers rather
//  than fades. The knee ramps contribution quadratically across a band around
//  the threshold instead, so the transition is continuous.
//
//  KARIS AVERAGE. A single very bright pixel (a specular glint on a curved
//  metal, a subpixel highlight) survives the downsample chain and becomes a
//  large soft blob that flickers as the camera moves -- the "firefly" artifact.
//  Weighting each tap by 1/(1+luma) before averaging bounds how much any one
//  pixel can contribute. It is not energy-preserving, deliberately: stability
//  matters more than exactness for something this diffuse.
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
layout(binding=0) uniform prefilter_params {
    // x = threshold, y = knee, z = 1/src width, w = 1/src height
    vec4 params;
};

layout(binding=0) uniform texture2D tex_src;
layout(binding=0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

float luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec3 karisTap(vec2 uv) {
    vec3 c = textureLod(sampler2D(tex_src, smp), uv, 0.0).rgb;
    return c / (1.0 + luma(c));
}

void main() {
    vec2 texel = params.zw;

    // Four bilinear taps at the corners of the source quad: with LINEAR
    // filtering each already averages 2x2, so this covers a 4x4 neighbourhood
    // for four fetches.
    vec3 sum = karisTap(v_uv + texel * vec2(-1.0, -1.0));
    sum += karisTap(v_uv + texel * vec2(1.0, -1.0));
    sum += karisTap(v_uv + texel * vec2(-1.0, 1.0));
    sum += karisTap(v_uv + texel * vec2(1.0, 1.0));
    vec3 color = sum * 0.25;

    float threshold = params.x;
    float knee = max(params.y, 1e-4);
    float b = max(max(color.r, color.g), color.b);

    // Quadratic ramp across [threshold - knee, threshold + knee], linear above.
    float soft = clamp(b - threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee);
    float contribution = max(soft, b - threshold) / max(b, 1e-4);

    frag_color = vec4(color * contribution, 1.0);
}
@end

@program bloom_prefilter vs fs
