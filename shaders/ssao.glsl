//------------------------------------------------------------------------------
//  shaders/ssao.glsl
//
//  Screen-space ambient occlusion from the deferred G-buffer.
//
//  WHY THIS EXISTS: image-based lighting arrives unoccluded. Every point on
//  every surface receives the full hemisphere of sky, so ambient reaches under
//  objects and into crevices at full strength, and things read as floating
//  rather than resting on the floor. A shadow map can't fix that -- the sky is
//  an area light covering the hemisphere, not a point, so there is no single
//  view to render depth from. What it needs is a visibility estimate, which is
//  what this computes.
//
//  METHOD: hemisphere sampling. Reconstruct world position from depth, scatter
//  samples through the hemisphere around the surface normal, project each back
//  to the screen, and compare its depth against what the G-buffer actually
//  recorded there. A sample buried behind recorded geometry is occluded.
//
//  Sample directions come from a Hammersley sequence rotated per pixel, rather
//  than a uniform-random kernel and a noise texture. Same decorrelation, no
//  extra binding, and the low-discrepancy sequence covers the hemisphere more
//  evenly at low sample counts -- which matters, because the count IS the
//  budget here. The rotation leaves a fine pattern that the blur pass removes.
//------------------------------------------------------------------------------

@vs vs
in vec2 pos;
in vec2 uv;

out vec2 v_uv;
out vec2 v_ndc;

void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
    v_ndc = pos;
    v_uv = uv;
}
@end

@fs fs
layout(binding=0) uniform ssao_params {
    mat4 inv_view_proj;
    mat4 view_proj;
    vec4 camera_pos;
    // x = radius (world units), y = intensity, z = depth bias,
    // w = 1 if the sampled render target reads top-left origin
    vec4 params;
    // x = sample count, y = range falloff, zw = 1 / resolution
    vec4 tuning;
};

layout(binding=0) uniform texture2D tex_depth;
layout(binding=1) uniform texture2D tex_normal;
layout(binding=0) uniform sampler smp;

in vec2 v_uv;
in vec2 v_ndc;
out vec4 frag_color;

const float PI = 3.14159265359;
const int MAX_SAMPLES = 32;

// World position from a UV's recorded depth.
vec3 worldAt(vec2 uv, out float raw_depth) {
    raw_depth = textureLod(sampler2D(tex_depth, smp), uv, 0.0).r;
    vec2 ndc = uv * 2.0 - 1.0;
    if (params.w > 0.5) ndc.y = -ndc.y;
    vec4 h = inv_view_proj * vec4(ndc, raw_depth, 1.0);
    return h.xyz / h.w;
}

// Low-discrepancy pair for sample i of n (van der Corput radical inverse).
vec2 hammersley(int i, float inv_n) {
    uint bits = uint(i);
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return vec2(float(i) * inv_n, float(bits) * 2.3283064365386963e-10);
}

// Interleaved gradient noise: cheap, and its error is high-frequency, which is
// exactly what a small blur removes cleanly.
float igNoise(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

void main() {
    vec4 nrm = textureLod(sampler2D(tex_normal, smp), v_uv, 0.0);
    // Background: fully unoccluded. Writing 1.0 rather than discarding keeps the
    // target defined everywhere, so the blur below never pulls in stale texels.
    if (nrm.a < 0.5) {
        frag_color = vec4(1.0);
        return;
    }

    float raw_depth;
    vec3 P = worldAt(v_uv, raw_depth);
    vec3 N = normalize(nrm.xyz);

    // Tangent basis around N. The branch avoids the degenerate cross product
    // when N happens to be the axis being crossed against.
    vec3 up = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, N));
    vec3 B = cross(N, T);

    int n = int(tuning.x);
    float inv_n = 1.0 / max(tuning.x, 1.0);
    float radius = params.x;
    float bias = params.z;

    // Per-pixel rotation, so neighbouring pixels sample different directions and
    // the error becomes noise the blur can average away instead of banding.
    float rot = igNoise(gl_FragCoord.xy) * 2.0 * PI;
    float cr = cos(rot);
    float sr = sin(rot);

    float occlusion = 0.0;
    for (int i = 0; i < MAX_SAMPLES; i++) {
        if (i >= n) break;

        vec2 u = hammersley(i, inv_n);

        // Cosine-weighted hemisphere: concentrates samples where the cosine term
        // in the ambient integral actually carries weight, instead of spending
        // them near the horizon where their contribution is almost zero.
        float phi = 2.0 * PI * u.x + rot;
        float cos_theta = sqrt(1.0 - u.y);
        float sin_theta = sqrt(u.y);
        vec3 dir = T * (cos(phi) * sin_theta) + B * (sin(phi) * sin_theta) + N * cos_theta;

        // Push samples toward the centre so the kernel is denser near the
        // surface, where occlusion detail lives.
        float scale = mix(0.1, 1.0, u.x * u.x);
        vec3 S = P + dir * radius * scale;

        // Project back to screen and read what the G-buffer really has there.
        vec4 clip = view_proj * vec4(S, 1.0);
        if (clip.w <= 0.0) continue;
        vec2 ndc = clip.xy / clip.w;
        vec2 uv = ndc * 0.5 + 0.5;
        if (params.w > 0.5) uv.y = 1.0 - uv.y;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) continue;

        float sample_depth;
        vec3 Q = worldAt(uv, sample_depth);

        // Occluded when the recorded surface sits nearer the camera than the
        // sample point does.
        float d_sample = distance(S, camera_pos.xyz);
        float d_scene = distance(Q, camera_pos.xyz);

        // Range check: a distant wall in front of this pixel is not occluding
        // it, it is simply a different surface. Without this, silhouettes gain
        // dark halos wherever a near object overlaps a far background.
        float range = smoothstep(0.0, 1.0, radius / max(abs(d_sample - d_scene), 1e-4));
        occlusion += (d_scene < d_sample - bias ? 1.0 : 0.0) * range;
    }

    float ao = 1.0 - (occlusion * inv_n) * params.y;
    // tuning.y sharpens or softens the falloff; 1.0 leaves it linear.
    frag_color = vec4(pow(clamp(ao, 0.0, 1.0), tuning.y));
}
@end

@program ssao vs fs
