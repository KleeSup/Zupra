//------------------------------------------------------------------------------
//  shaders/ssao_blur.glsl
//
//  Separable, depth-aware blur for the SSAO buffer.
//
//  SEPARABLE: run twice, horizontally then vertically. A radius-2 kernel is 5
//  taps per pass instead of 25 for the equivalent 5x5 box, and the saving grows
//  quadratically with radius. Gaussian-style blurs are separable exactly; this
//  one is only approximately so once the depth weights are applied, but the
//  error is far below the noise it is removing.
//
//  DEPTH-AWARE: averaging across a silhouette drags occlusion from a foreground
//  object onto the background behind it, drawing a dark halo around every
//  object -- the most recognisable SSAO artifact there is.
//
//  Rejection uses LINEAR view depth with a RELATIVE threshold, not raw clip
//  depth. Raw depth is wildly non-linear: near the camera it changes so fast
//  between adjacent pixels that an absolute threshold rejects nearly every
//  neighbour, the blur stops doing anything, and the per-pixel noise survives as
//  crawling speckle. Relative to the centre pixel's own depth, the same
//  threshold means the same thing at every distance.
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
layout(binding=0) uniform blur_params {
    // xy = step in uv per tap (one axis zero: that is what makes it separable),
    // z  = kernel radius in taps, w = relative depth tolerance.
    vec4 params;
    // xy = the A and B of z_view = B / (depth - A).
    vec4 depth_lin;
};

layout(binding=0) uniform texture2D tex_ao;
layout(binding=1) uniform texture2D tex_depth;
layout(binding=0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

float viewZ(vec2 uv) {
    float raw = textureLod(sampler2D(tex_depth, smp), uv, 0.0).r;
    return depth_lin.y / (raw - depth_lin.x);
}

void main() {
    float center_z = viewZ(v_uv);
    int radius = int(params.z);

    // Tolerance scales with distance: a 5cm depth step matters up close and is
    // noise at fifty metres.
    float tolerance = center_z * params.w;

    float sum = 0.0;
    float weight_sum = 0.0;

    for (int i = -8; i <= 8; i++) {
        if (i < -radius || i > radius) continue;

        vec2 uv = v_uv + params.xy * float(i);
        float z = viewZ(uv);

        // Binary accept/reject rather than a falloff: a tap is either on this
        // surface or it is not, and a soft weight just leaks a fraction of the
        // halo back in.
        float w = abs(z - center_z) < tolerance ? 1.0 : 0.0;

        sum += textureLod(sampler2D(tex_ao, smp), uv, 0.0).r * w;
        weight_sum += w;
    }

    frag_color = vec4(sum / max(weight_sum, 1.0));
}
@end

@program ssao_blur vs fs
