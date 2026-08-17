//------------------------------------------------------------------------------
//  shaders/xegtao_denoise.glsl
//
//  Edge-aware denoiser for the XeGTAO output.
//
//  The main pass rotates its slice directions per pixel, which turns what would
//  be visible banding into high-frequency noise. That trade only pays off if
//  something averages the noise back out, and this is it.
//
//  DEPTH-AWARE, not a box blur: averaging across a silhouette drags occlusion
//  from a foreground surface onto the background behind it, which draws a dark
//  halo around every object and is the most recognisable AO artifact there is.
//
//  Rejection uses LINEAR view depth from the pyramid, with a threshold relative
//  to the centre pixel's own depth. Raw clip depth is so non-linear that an
//  absolute threshold rejects nearly every neighbour close to the camera, the
//  filter stops working, and the noise survives as crawling speckle.
//
//  Separable: run once per axis. A radius-3 kernel is 7 taps per pass instead of
//  49 for the equivalent square, and the saving grows quadratically.
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
layout(binding=0) uniform denoise_params {
    // xy = step in uv per tap, one axis zero, which is what makes it separable.
    // z = kernel radius in taps, w = relative depth tolerance.
    vec4 params;
};

layout(binding=0) uniform texture2D tex_ao;
layout(binding=1) uniform texture2D tex_depth;
layout(binding=0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    float center_z = textureLod(sampler2D(tex_depth, smp), v_uv, 0.0).r;
    int radius = int(params.z);

    // Tolerance scales with distance: a five centimetre step matters up close
    // and is noise at fifty metres.
    float tolerance = center_z * params.w;

    float sum = 0.0;
    float weight_sum = 0.0;

    for (int i = -8; i <= 8; i++) {
        if (i < -radius || i > radius) continue;

        vec2 uv = v_uv + params.xy * float(i);
        float z = textureLod(sampler2D(tex_depth, smp), uv, 0.0).r;

        // Binary accept or reject rather than a soft falloff. A tap is either on
        // this surface or it is not, and a partial weight just leaks a fraction
        // of the halo back in.
        float w = (abs(z - center_z) < tolerance) ? 1.0 : 0.0;

        sum += textureLod(sampler2D(tex_ao, smp), uv, 0.0).r * w;
        weight_sum += w;
    }

    frag_color = vec4(sum / max(weight_sum, 1.0));
}
@end

@program xegtao_denoise vs fs
