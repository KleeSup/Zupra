//------------------------------------------------------------------------------
//  shaders/sprite_fancy.glsl
//
//  A DROP-IN custom sprite shader demonstrating SpriteBatch customization.
//
//  Preserves the exact SpriteBatch contract (zero batcher changes needed):
//    * vertex inputs at locations 0/1/2 (pos, uv, color0) = VertexLayout.sprite
//    * vertex uniform block `vs_params { mat4 mvp; }` (uploaded by the batch)
//    * fragment texture `tex` + sampler `smp` at binding 0
//
//  Effect: a radial RAINBOW wheel masked by the texture's alpha, with the
//  texture luminance preserved so detail still reads. Uses only uv + texture +
//  per-vertex tint, so it needs no extra uniforms. As the demo spins the sprite
//  on the CPU, the rainbow wheel spins with it. (A *flowing* time-based rainbow
//  would need an fs uniform block + the batch uploading fragment uniforms.)
//------------------------------------------------------------------------------

@vs vs
layout(binding=0) uniform vs_params {
    mat4 mvp;
};

in vec2 pos;
in vec2 uv;
in vec4 color0;

out vec2 v_uv;
out vec4 v_color;

void main() {
    gl_Position = mvp * vec4(pos, 0.0, 1.0);
    v_uv = uv;
    v_color = color0;
}
@end

@fs fs
layout(binding=0) uniform texture2D tex;
layout(binding=0) uniform sampler smp;

in vec2 v_uv;
in vec4 v_color;

out vec4 frag_color;

// IQ's hue->rgb (h,s,v in 0..1).
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    // Hue from the angle around the sprite center -> a full rainbow wheel.
    vec2 c = v_uv - vec2(0.5);
    float hue = atan(c.y, c.x) / 6.2831853 + 0.5;
    vec3 rainbow = hsv2rgb(vec3(hue, 0.9, 1.0));

    // Keep the texture's shape (alpha mask) and shading (luminance).
    vec4 texel = texture(sampler2D(tex, smp), v_uv);
    float lum = dot(texel.rgb, vec3(0.299, 0.587, 0.114));
    vec3 col = rainbow * mix(0.45, 1.0, lum);

    // Per-vertex tint still applies; alpha from the texture for clean blending.
    frag_color = vec4(col, texel.a) * v_color;
}
@end

@program sprite_fancy vs fs
