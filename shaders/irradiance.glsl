//------------------------------------------------------------------------------
//  shaders/irradiance.glsl
//
//  Diffuse irradiance convolution. For each output texel, reconstruct the world
//  direction N from this face's inverse view-projection, build a tangent basis,
//  and cosine-weight-integrate the source environment cubemap over the
//  hemisphere around N. The result (a small cubemap) is the diffuse ambient a
//  surface facing N receives from the whole environment.
//
//  conv_params (binding 0): mat4 inv_view_proj  -- for the face being rendered
//  env_cube (binding 0): the source environment cubemap
//------------------------------------------------------------------------------

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
layout(binding=0) uniform conv_params {
    mat4 inv_view_proj;
};

in vec2 v_ndc;
out vec4 frag_color;

const float PI = 3.14159265359;

void main() {
    // Output direction for this cube-face pixel (camera at origin).
    vec4 world = inv_view_proj * vec4(v_ndc, 1.0, 1.0);
    vec3 N = normalize(world.xyz / world.w);

    // Tangent basis around N.
    vec3 up = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
    vec3 right = normalize(cross(up, N));
    up = normalize(cross(N, right));

    vec3 irradiance = vec3(0.0);
    float nrSamples = 0.0;
    const float sampleDelta = 0.025;

    for (float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta) {
        for (float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta) {
            // spherical -> tangent -> world
            vec3 tangent = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            vec3 sampleVec = tangent.x * right + tangent.y * up + tangent.z * N;
            // cos(theta): cosine weight; sin(theta): solid-angle (smaller near pole)
            irradiance += texture(samplerCube(env_cube, smp), sampleVec).rgb * cos(theta) * sin(theta);
            nrSamples += 1.0;
        }
    }
    irradiance = PI * irradiance / nrSamples;
    frag_color = vec4(irradiance, 1.0);
}
@end

@program irradiance vs fs
