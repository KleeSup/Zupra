//------------------------------------------------------------------------------
//  shaders/sky_equirect.glsl
//
//  Samples an equirectangular (lat-long) HDR environment by world direction.
//
//  Used twice with the same code: as the screen background at full resolution,
//  and to fill the six faces of the IBL env cubemap. The only difference is the
//  inverse view-projection handed in -- the camera's for the background, each
//  cube face's for the bake.
//
//  Reconstructs the view ray the same way sky.glsl does: unproject the far-plane
//  clip position and subtract the camera. That keeps the two interchangeable
//  from the renderer's side.
//------------------------------------------------------------------------------

@vs vs
in vec2 pos;
in vec2 uv;

out vec2 v_clip;

void main() {
    // z = 1 puts the background on the far plane, so depth testing lets any
    // geometry occlude it while depth writes stay off.
    gl_Position = vec4(pos, 1.0, 1.0);

    // The unprojection happens in the fragment stage, so only the clip position
    // travels. shdc rejects a uniform block declared in two stages, and keeping
    // the block fragment-only is what sky.glsl already does -- worth matching so
    // the two stay interchangeable from the renderer's side.
    v_clip = pos;

    // Keep `uv` referenced so shdc doesn't strip it and leave a vertex input
    // signature that no longer matches VertexLayout.fullscreen.
    gl_Position.x += uv.x * 0.0;
}
@end

@fs fs
layout(binding=0) uniform equirect_params {
    mat4 inv_view_proj;
    vec4 camera_pos;
    vec4 params;      // x = intensity
};

layout(binding=0) uniform texture2D equirect;
layout(binding=0) uniform sampler smp;

in vec2 v_clip;
out vec4 frag_color;

const float PI = 3.14159265359;

void main() {
    // Row-major rows arrive as four vec4s and mat4() consumes them as columns,
    // so the matrix is already transposed and M * v is the row-vector transform.
    vec4 world = inv_view_proj * vec4(v_clip, 1.0, 1.0);
    vec3 d = normalize(world.xyz / world.w - camera_pos.xyz);

    // Direction -> lat-long. u wraps with longitude around Y; v runs from the
    // top of the image (+Y) down, matching how the file is stored: stb loads
    // row 0 first, and row 0 of an equirectangular map is straight up.
    vec2 uv = vec2(
        atan(d.z, d.x) / (2.0 * PI) + 0.5,
        acos(clamp(d.y, -1.0, 1.0)) / PI
    );

    // textureLod with an explicit level, not texture(): u jumps by a full turn
    // across the seam directly behind the camera, and an implicit derivative
    // there would read that jump as an enormous rate of change and collapse to
    // the smallest mip, drawing a visible line down the sky.
    vec3 radiance = textureLod(sampler2D(equirect, smp), uv, 0.0).rgb * params.x;

    frag_color = vec4(radiance, 1.0);
}
@end

@program sky_equirect vs fs
