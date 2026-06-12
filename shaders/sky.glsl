//------------------------------------------------------------------------------
//  shaders/sky.glsl
//
//  Procedural skybox. A fullscreen pass: reconstruct the world-space view ray
//  per pixel from the inverse view-projection, then shade a gradient sky (horizon
//  -> zenith) plus a sun disk. Outputs at the far plane (clip z = 1) so the depth
//  test against the opaque depth buffer lets geometry occlude it — only
//  background pixels get sky.
//
//  sky_params (binding 0):
//    mat4 inv_view_proj   -- inverse(camera.viewProjection()), uploaded directly
//    vec4 camera_pos      -- xyz
//    vec4 sky_top         -- zenith color (rgb)
//    vec4 sky_horizon     -- horizon color (rgb)
//    vec4 sun_dir         -- xyz direction-to-sun, w = sun sharpness (exponent)
//    vec4 sun_color       -- rgb, w = intensity
//------------------------------------------------------------------------------

@vs vs
in vec2 pos;
in vec2 uv;
out vec2 v_ndc;
void main() {
    // z = 1 (far plane) so the skybox sits behind all geometry under LEQUAL.
    gl_Position = vec4(pos, 1.0, 1.0);
    v_ndc = pos; // fullscreen-triangle pos IS clip/NDC xy
}
@end

@fs fs
layout(binding=0) uniform sky_params {
    mat4 inv_view_proj;
    vec4 camera_pos;
    vec4 sky_top;
    vec4 sky_horizon;
    vec4 sun_dir;
    vec4 sun_color;
};

in vec2 v_ndc;
out vec4 frag_color;

void main() {
    // Reconstruct the world-space ray through this pixel.
    vec4 world = inv_view_proj * vec4(v_ndc, 1.0, 1.0);
    vec3 ray = normalize(world.xyz / world.w - camera_pos.xyz);

    // Vertical gradient: horizon at ray.y ~ 0, zenith at ray.y ~ 1.
    float h = clamp(ray.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 col = mix(sky_horizon.rgb, sky_top.rgb, pow(h, 0.6));

    // Sun disk + glow.
    float s = max(dot(ray, normalize(sun_dir.xyz)), 0.0);
    col += sun_color.rgb * sun_color.w * pow(s, max(sun_dir.w, 1.0));

    frag_color = vec4(col, 1.0);
}
@end

@program sky vs fs
