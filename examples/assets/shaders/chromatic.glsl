//------------------------------------------------------------------------------
//  shaders/chromatic.glsl
//
//  EXAMPLE USER POST-EFFECT — radial chromatic aberration.
//
//  This is the template for any custom post effect. Copy it, change the fragment
//  main, keep everything else. The declarations below are the CONTRACT PostChain
//  binds against (posprocess.zig: view_input = 0, view_depth = 1, smp_input = 0);
//  an effect that renames or reorders them won't receive its inputs.
//
//  Physically this imitates a lens failing to focus all wavelengths at the same
//  point: the error grows with distance from the optical axis, so the offset is
//  scaled by radius. Sampling R and B displaced in opposite directions along that
//  radius gives the red/blue fringing that's absent at the centre and strongest
//  in the corners.
//
//  Runs at .ldr (after tonemap) — it's a display-referred look, not a light
//  transport effect. Applying it in HDR would fringe values that later get
//  compressed by the tonemapper, producing an inconsistent result.
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
layout(binding=0) uniform texture2D u_input; // previous stage's colour
layout(binding=0) uniform sampler u_smp;     // linear, clamped to edge

layout(binding=0) uniform fs_params {
    // x = strength (0.002 subtle, 0.01 strong, 0.05 broken-lens)
    // y = falloff exponent (1 = linear from centre, 2 = corners only)
    // z, w = unused
    vec4 params;
};

in vec2 v_uv;
out vec4 frag_color;

void main() {
    float strength = params.x;
    float falloff = max(params.y, 1.0);

    vec2 to_centre = v_uv - vec2(0.5);
    float radius = length(to_centre) * 2.0; // 0 at centre, ~1 at edges

    // Offset grows with radius, so the centre of frame stays sharp.
    vec2 offset = to_centre * pow(radius, falloff) * strength;

    // Green stays put and defines the image; red and blue shift oppositely.
    float r = textureLod(sampler2D(u_input, u_smp), v_uv - offset, 0.0).r;
    float g = textureLod(sampler2D(u_input, u_smp), v_uv, 0.0).g;
    float b = textureLod(sampler2D(u_input, u_smp), v_uv + offset, 0.0).b;

    frag_color = vec4(r, g, b, 1.0);
}
@end

@program chromatic vs fs
