//------------------------------------------------------------------------------
//  shaders/ssao.glsl
//
//  Screen-space ambient occlusion, computed in VIEW SPACE.
//
//  WHY THIS EXISTS: image-based lighting arrives unoccluded. Every point
//  receives the full hemisphere of sky regardless of what stands next to it, so
//  ambient reaches under objects and into corners at full strength and geometry
//  reads as floating. A shadow map cannot fix it -- the sky is an area source
//  covering the hemisphere, so there is no single view to render depth from.
//
//  WHY VIEW SPACE: the naive formulation reconstructs a world position per
//  sample (a mat4 multiply and a perspective divide each way) and compares
//  radial distances. View space collapses all of that into scalar arithmetic,
//  because for a perspective projection the mapping between a view-space point
//  and its screen position is two multiplies:
//
//      ndc.x = S.x * m00 / S.z          S.x = ndc.x * S.z / m00
//      ndc.y = S.y * m11 / S.z          S.y = ndc.y * S.z / m11
//
//  and view depth comes straight from the raw depth buffer as z = B/(d - A).
//  No matrices in the inner loop at all: one texture fetch, one reciprocal and
//  a few multiplies per sample. That difference is most of why this can run at
//  full resolution.
//
//  Sample directions come from a Hammersley sequence rotated per pixel rather
//  than a random kernel plus a noise texture -- same decorrelation, one fewer
//  binding, better hemisphere coverage at low sample counts. The rotation leaves
//  fine noise that the separable blur removes.
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
    // World -> view, for rotating the G-buffer's world-space normal.
    mat4 view;
    // x = proj[0][0], y = proj[1][1], z = 1/x, w = 1/y.
    vec4 proj_xy;
    // x, y = the A and B of z_view = B / (depth - A). z = 1 when the render
    // target reads top-left first. w = sample count.
    vec4 depth_lin;
    // x = radius (view units), y = intensity, z = bias, w = falloff exponent.
    vec4 params;
};

layout(binding=0) uniform texture2D tex_depth;
layout(binding=1) uniform texture2D tex_normal;
layout(binding=0) uniform sampler smp;

in vec2 v_uv;
in vec2 v_ndc;
out vec4 frag_color;

const float PI = 3.14159265359;

// Raw [0,1] clip depth -> positive view-space z.
float viewZ(float raw) {
    return depth_lin.y / (raw - depth_lin.x);
}

float sampleViewZ(vec2 uv) {
    return viewZ(textureLod(sampler2D(tex_depth, smp), uv, 0.0).r);
}

// NDC + view depth -> view-space position.
vec3 viewPos(vec2 ndc, float z) {
    return vec3(ndc.x * z * proj_xy.z, ndc.y * z * proj_xy.w, z);
}

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
// exactly what the blur removes.
float igNoise(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

void main() {
    vec4 nrm = textureLod(sampler2D(tex_normal, smp), v_uv, 0.0);
    // Background: fully unoccluded. Writing 1.0 rather than discarding keeps the
    // target defined everywhere, so the blur never pulls in undefined texels.
    if (nrm.a < 0.5) {
        frag_color = vec4(1.0);
        return;
    }

    float z = sampleViewZ(v_uv);
    vec3 P = viewPos(v_ndc, z);
    // Rows arrive as columns, so view * v is the row-vector transform. w = 0
    // because a normal is a direction and translation must not apply.
    vec3 N = normalize((view * vec4(normalize(nrm.xyz), 0.0)).xyz);

    int n = int(depth_lin.w);
    float inv_n = 1.0 / max(depth_lin.w, 1.0);
    float radius = params.x;
    float bias = params.z;
    bool flip = depth_lin.z > 0.5;

    // Tangent basis. The branch avoids a degenerate cross product when N is
    // parallel to the axis being crossed against.
    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, N));
    vec3 B = cross(N, T);

    // Per-pixel rotation, so neighbours sample different directions and the
    // error becomes noise the blur can average rather than visible banding.
    float rot = igNoise(gl_FragCoord.xy) * 2.0 * PI;

    float occlusion = 0.0;
    // Exact iteration count. A fixed upper bound with a break is unrollable, and
    // FXC will happily unroll it to the maximum regardless of what n actually is.
    for (int i = 0; i < n; i++) {
        vec2 u = hammersley(i, inv_n);

        // Cosine-weighted hemisphere: concentrates samples where the cosine term
        // in the ambient integral carries weight, instead of near the horizon
        // where the contribution is almost nothing.
        float phi = 2.0 * PI * u.x + rot;
        float cos_theta = sqrt(1.0 - u.y);
        float sin_theta = sqrt(u.y);
        vec3 dir = T * (cos(phi) * sin_theta) + B * (sin(phi) * sin_theta) + N * cos_theta;

        // Bias the kernel toward the centre, where occlusion detail lives.
        float scale = mix(0.1, 1.0, u.x * u.x);
        vec3 S = P + dir * (radius * scale);

        // Behind the eye: no valid projection.
        if (S.z <= 0.0001) continue;

        // Project. Two multiplies and a reciprocal -- no matrix.
        float inv_z = 1.0 / S.z;
        vec2 ndc = vec2(S.x * proj_xy.x * inv_z, S.y * proj_xy.y * inv_z);
        if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) continue;

        vec2 uv = ndc * 0.5 + 0.5;
        if (flip) uv.y = 1.0 - uv.y;

        float scene_z = sampleViewZ(uv);

        // Occluded when the recorded surface sits nearer the eye than the sample
        // point does.
        //
        // The range term keeps a distant wall from occluding a near pixel it
        // merely happens to sit behind on screen -- that is a different surface,
        // not an occluder. Without it every silhouette gains a dark halo.
        float range = clamp(radius / max(abs(z - scene_z), 1e-4), 0.0, 1.0);
        occlusion += step(scene_z, S.z - bias) * range;
    }

    float ao = 1.0 - (occlusion * inv_n) * params.y;
    frag_color = vec4(pow(clamp(ao, 0.0, 1.0), params.w));
}
@end

@program ssao vs fs
