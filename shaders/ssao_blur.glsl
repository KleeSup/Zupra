//------------------------------------------------------------------------------
//  shaders/ssao_blur.glsl
//
//  Edge-aware blur for the SSAO buffer.
//
//  The AO pass rotates its sample directions per pixel, which turns what would
//  be visible banding into high-frequency noise. That trade is only worth
//  anything if the noise then gets averaged away, which is this pass.
//
//  DEPTH-AWARE, not a plain box blur: averaging across a silhouette drags
//  occlusion from a foreground object onto the background behind it, which shows
//  up as a dark halo tracing every object edge -- the single most recognisable
//  SSAO artifact. Weighting each tap by how close its depth is to the centre
//  pixel's keeps the filter inside one surface.
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
    // xy = 1 / resolution, z = depth rejection sharpness, w = kernel radius
    vec4 params;
};

layout(binding=0) uniform texture2D tex_ao;
layout(binding=1) uniform texture2D tex_depth;
layout(binding=0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    float center_depth = textureLod(sampler2D(tex_depth, smp), v_uv, 0.0).r;
    int radius = int(params.w);

    float sum = 0.0;
    float weight_sum = 0.0;

    for (int y = -4; y <= 4; y++) {
        for (int x = -4; x <= 4; x++) {
            if (x < -radius || x > radius || y < -radius || y > radius) continue;

            vec2 offset = vec2(float(x), float(y)) * params.xy;
            vec2 uv = v_uv + offset;

            float d = textureLod(sampler2D(tex_depth, smp), uv, 0.0).r;
            // Depth difference in raw [0,1] clip units. Non-linear, so the
            // rejection naturally tightens with distance -- which is what you
            // want, since a fixed world-space threshold would be far too
            // permissive near the camera and far too strict at range.
            float w = exp(-abs(d - center_depth) * params.z);

            sum += textureLod(sampler2D(tex_ao, smp), uv, 0.0).r * w;
            weight_sum += w;
        }
    }

    frag_color = vec4(sum / max(weight_sum, 1e-4));
}
@end

@program ssao_blur vs fs
