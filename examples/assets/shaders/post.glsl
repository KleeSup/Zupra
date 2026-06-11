//------------------------------------------------------------------------------
//  shaders/post.glsl
//
//  Full-screen post-processing shader. Drop-in on the SpriteBatch contract
//  (same vertex inputs + vs_params { mvp }), so it runs by drawing the
//  framebuffer's color texture as a fullscreen sprite through beginEx with this
//  as the custom shader. `tex` is the rendered scene; the fragment stage
//  applies a CRT look across the whole image.
//
//  fs_params (binding 1): params.x = time (seconds), for animation.
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
layout(binding=1) uniform fs_params {
    vec4 params; // x = time
};

in vec2 v_uv;
in vec4 v_color;

out vec4 frag_color;

void main() {
    float time = params.x;
    vec2 uv = v_uv;

    vec2 c = uv - vec2(0.5);
    float r = length(c);
    vec2 dir = (r > 0.0001) ? c / r : vec2(0.0);

    // Radial chromatic aberration (subtle — this is the whole scene).
    float aberr = 0.0035 * r;
    float cr = texture(sampler2D(tex, smp), uv - dir * aberr).r;
    float cg = texture(sampler2D(tex, smp), uv).g;
    float cb = texture(sampler2D(tex, smp), uv + dir * aberr).b;
    vec3 col = vec3(cr, cg, cb);

    // Scrolling scanlines.
    float scan = 0.92 + 0.08 * sin(uv.y * 800.0 + time * 6.0);
    col *= scan;

    // Vignette.
    float vig = smoothstep(0.95, 0.25, r);
    col *= mix(0.45, 1.0, vig);

    frag_color = vec4(col, 1.0) * v_color;
}
@end

@program post vs fs
