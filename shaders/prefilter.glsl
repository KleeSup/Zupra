//------------------------------------------------------------------------------
//  shaders/prefilter.glsl
//
//  Specular IBL prefilter. For each output direction R (= N = V, the split-sum
//  assumption), GGX-importance-sample the environment cubemap and accumulate the
//  reflected radiance for ONE roughness level. Rendered once per mip: mip 0 =
//  roughness 0 (sharp mirror), higher mips = rougher (blurrier) reflections.
//
//  pre_params (binding 0):
//    mat4 inv_view_proj  -- for the face being rendered (reconstructs R)
//    vec4 params         -- x = roughness for this mip
//  env_cube (binding 0): the source environment cubemap
//------------------------------------------------------------------------------

@include pbr_lib.glsl.inc

@vs vs
in vec2 pos;
in vec2 uv;
out vec2 v_ndc;
void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
    v_ndc = pos;
}
@end

@fs fs
layout(binding=0) uniform textureCube env_cube;
layout(binding=0) uniform sampler smp;
layout(binding=0) uniform pre_params {
    mat4 inv_view_proj;
    vec4 params; // x = roughness
};

in vec2 v_ndc;
out vec4 frag_color;

@include_block pbr_hammersley

const uint SAMPLE_COUNT = 1024u;

// GGX importance sample: a half-vector around N for the given roughness.
vec3 importanceSampleGGX(vec2 Xi, vec3 N, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    vec3 H = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

void main() {
    vec4 world = inv_view_proj * vec4(v_ndc, 1.0, 1.0);
    vec3 N = normalize(world.xyz / world.w);
    vec3 R = N;
    vec3 V = N; // split-sum assumption: V = R = N

    float roughness = params.x;

    vec3 prefiltered = vec3(0.0);
    float totalWeight = 0.0;

    for (uint i = 0u; i < SAMPLE_COUNT; i++) {
        vec2 Xi = hammersley(i, SAMPLE_COUNT);
        vec3 H = importanceSampleGGX(Xi, N, roughness);
        vec3 L = normalize(2.0 * dot(V, H) * H - V);

        float ndl = max(dot(N, L), 0.0);
        if (ndl > 0.0) {
            prefiltered += texture(samplerCube(env_cube, smp), L).rgb * ndl;
            totalWeight += ndl;
        }
    }
    prefiltered = prefiltered / max(totalWeight, 0.001);
    frag_color = vec4(prefiltered, 1.0);
}
@end

@program prefilter vs fs
