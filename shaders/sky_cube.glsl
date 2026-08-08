//------------------------------------------------------------------------------
//  shaders/sky_cube.glsl
//
//  Samples a cubemap environment by world direction.
//
//  Identical to sky_equirect.glsl except for the lookup itself: a cubemap is
//  indexed by direction directly, so the lat-long conversion disappears and with
//  it the seam behind the camera that made the explicit mip level necessary
//  there. Kept as a separate program rather than a branch because the texture
//  type differs, and a shader can't be polymorphic over that.
//
//  Uniform block lives in the fragment stage only -- shdc rejects the same block
//  declared in two stages, so the vertex stage passes clip position through and
//  the unprojection happens below.
//------------------------------------------------------------------------------

@vs vs
in vec2 pos;
in vec2 uv;

out vec2 v_clip;

void main() {
    // z = 1 puts the background on the far plane: depth testing lets geometry
    // occlude it, depth writes stay off.
    gl_Position = vec4(pos, 1.0, 1.0);
    v_clip = pos;

    // Keep `uv` referenced so shdc doesn't strip it and leave a vertex input
    // signature that no longer matches VertexLayout.fullscreen.
    gl_Position.x += uv.x * 0.0;
}
@end

@fs fs
layout(binding=0) uniform cube_params {
    mat4 inv_view_proj;
    vec4 camera_pos;
    vec4 params;      // x = intensity
};

layout(binding=0) uniform textureCube env_map;
layout(binding=0) uniform sampler smp;

in vec2 v_clip;
out vec4 frag_color;

void main() {
    // Row-major rows arrive as four vec4s and mat4() consumes them as columns,
    // so the matrix is already transposed and M * v is the row-vector transform.
    vec4 world = inv_view_proj * vec4(v_clip, 1.0, 1.0);
    vec3 d = normalize(world.xyz / world.w - camera_pos.xyz);

    frag_color = vec4(texture(samplerCube(env_map, smp), d).rgb * params.x, 1.0);
}
@end

@program sky_cube vs fs
