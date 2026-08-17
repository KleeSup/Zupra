//------------------------------------------------------------------------------
//  shaders/xegtao_prefilter.glsl
//
//  Converts the depth buffer to view-space Z, and generates one level of the
//  depth pyramid the main pass samples from.
//
//  WHY A PYRAMID AT ALL. This is the piece our own GTAO lacked, and the reason
//  it kept failing. A horizon search wants to cover a world-space radius, which
//  projects to a wildly varying number of pixels: hundreds when close to a
//  surface, a fraction of one when far away. Marching that in pixels and
//  clamping the range silently changes what the effect measures depending on
//  where the camera stands. Marching it in world units instead costs an
//  unbounded number of taps up close.
//
//  A pyramid removes the dilemma. A sample far from the pixel reads a coarse
//  level, where one texel already covers the ground several fine taps would
//  have. The search covers the full radius with a fixed, small tap count from
//  every distance, so the measurement stops depending on the viewpoint.
//
//  THE DOWNSAMPLE IS NOT AN AVERAGE. A plain average of four depths across a
//  silhouette produces a value describing no real surface, and the main pass
//  would treat that phantom as an occluder. The filter below weights toward the
//  FARTHEST of the four, so thin near geometry cannot dominate a coarse level.
//  Losing a thin occluder at coarse scale is the right trade: it contributes
//  little occlusion, whereas a phantom surface contributes a great deal of
//  wrong occlusion.
//
//  Ported from the technique in Intel's XeGTAO (Jimenez et al. GTAO, as
//  implemented by Intel), adapted to this renderer's conventions: view-space Z
//  is positive, matrices are row-vector, and the origin flip is explicit.
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
    // x, y = the A and B of z_view = B / (depth - A).
    // z = 1 when this is level 0 and the source is the raw depth buffer,
    //     0 when downsampling an already-linear level. w unused.
    vec4 params;
    // xy = 1 / source resolution, z = source mip to read, w = effect radius.
    vec4 source;
    // x = falloff multiplier, y = falloff addend, for the outlier rejection.
    vec4 falloff;
};

layout(binding=0) uniform texture2D tex_source;
layout(binding=0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

float rawToView(float raw) {
    return params.y / (raw - params.x);
}

void main() {
    vec2 texel = source.xy;
    float src_mip = source.z;

    if (params.z > 0.5) {
        // Level 0: read the hardware depth buffer and linearise it. A single
        // tap, since this level must match the depth buffer exactly for the main
        // pass to reconstruct positions correctly.
        //
        // NO ORIGIN FLIP HERE. v_uv comes from the fullscreen triangle, which
        // already bakes the backend's origin convention into its uv attribute,
        // so it addresses any render target correctly as it stands. The depth
        // buffer is such a target. Flipping again mirrors the whole pyramid, and
        // since the main pass converts uv back to NDC with the flip applied, the
        // result is an AO buffer that is upside down relative to the scene.
        frag_color = vec4(rawToView(textureLod(sampler2D(tex_source, smp), v_uv, 0.0).r), 0.0, 0.0, 1.0);
        return;
    }

    // Downsample: four taps from the level above, at its texel centres.
    vec2 base = v_uv - texel * 0.5;
    float d0 = textureLod(sampler2D(tex_source, smp), base + texel * vec2(0.0, 0.0), src_mip).r;
    float d1 = textureLod(sampler2D(tex_source, smp), base + texel * vec2(1.0, 0.0), src_mip).r;
    float d2 = textureLod(sampler2D(tex_source, smp), base + texel * vec2(0.0, 1.0), src_mip).r;
    float d3 = textureLod(sampler2D(tex_source, smp), base + texel * vec2(1.0, 1.0), src_mip).r;

    // Weight each tap by how close it is to the FARTHEST of the four, so a near
    // sliver of geometry cannot pull a coarse texel toward itself and become a
    // phantom occluder for everything behind it.
    float far_d = max(max(d0, d1), max(d2, d3));
    float w0 = clamp((far_d - d0) * falloff.x + falloff.y, 0.0, 1.0);
    float w1 = clamp((far_d - d1) * falloff.x + falloff.y, 0.0, 1.0);
    float w2 = clamp((far_d - d2) * falloff.x + falloff.y, 0.0, 1.0);
    float w3 = clamp((far_d - d3) * falloff.x + falloff.y, 0.0, 1.0);

    float wsum = w0 + w1 + w2 + w3;
    float result = (wsum > 1e-6)
        ? (w0 * d0 + w1 * d1 + w2 * d2 + w3 * d3) / wsum
        : far_d;

    frag_color = vec4(result, 0.0, 0.0, 1.0);
}
@end

@program xegtao_prefilter vs fs
