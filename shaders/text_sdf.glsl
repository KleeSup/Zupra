//------------------------------------------------------------------------------
//  shaders/text_sdf.glsl
//
//  Signed-distance-field text shader. Drop-in on the SpriteBatch contract
//  (same vertex inputs + vs_params { mvp }), so text rides the batcher via
//  beginEx(.., custom_shader). The atlas is a single-channel (R8) distance
//  field: tex.r encodes distance, with the glyph edge at ~0.5.
//
//  Crispness at ANY size comes from fwidth(dist): it measures how fast the
//  distance changes in screen space, so the anti-alias band is always ~1 pixel
//  wide whether the text is 12px UI or a giant 3D banner. That's what makes SDF
//  "scalable" and 3D-ready — no re-baking per size.
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

void main() {
    // Distance from the glyph edge (edge baked at onedge_value/255 ~= 0.5).
    float dist = texture(sampler2D(tex, smp), v_uv).r;

    // Screen-space smoothing width: ~1px AA band regardless of text size.
    float w = max(fwidth(dist), 0.0001);
    float alpha = smoothstep(0.5 - w, 0.5 + w, dist);

    // Tint from the per-vertex color; coverage from the SDF.
    frag_color = vec4(v_color.rgb, v_color.a * alpha);
}
@end

@program text_sdf vs fs
