//------------------------------------------------------------------------------
//  shaders/text_sdf.glsl
//
//  Signed-distance-field text shader. Drop-in on the SpriteBatch contract
//  (same vertex inputs + vs_params { mvp }), so text rides the batcher via
//  beginEx(.., custom_shader). The atlas is a single-channel (R8) distance
//  field: tex.r encodes distance, with the glyph edge at onedge/255.
//
//  Crispness at any size comes from converting the field's distance into SCREEN
//  PIXELS and using that to set coverage, so the anti-alias band is always about
//  one pixel wide whether the text is 12px UI or a giant 3D banner. That is what
//  makes SDF scalable with no re-baking per size.
//
//  DERIVATIVES ARE TAKEN OF THE UV, NOT OF THE SAMPLED DISTANCE. This matters
//  more than it looks. GPUs evaluate derivatives per 2x2 pixel quad, so
//  fwidth(dist) at a glyph edge mixes a pixel sampling inside the field with one
//  sampling well outside it, and hands the whole quad a meaningless rate of
//  change. The visible result is a hard crease or a smear a pixel or two long on
//  some glyph edges and not others -- which glyphs depends on their subpixel
//  phase, so the artifact appears to wander as text changes length.
//
//  v_uv is linear across the triangle, so its derivative is exact everywhere and
//  the same everywhere on a glyph. This is the formulation msdfgen documents,
//  and the reason it is written that way.
//
//  BAKE CONTRACT with font.zig (sdf_padding / sdf_onedge / sdf_pixel_dist_scale,
//  near the top of that file). Change them there and these must follow:
//      value = onedge + distance_in_texels * pixel_dist_scale
//  with onedge = 128 and pixel_dist_scale = onedge / padding = 128 / 4 = 32.
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

// Mirrors font.zig's bake constants.
const float SDF_ONEDGE = 128.0 / 255.0;
// Sampled units -> texels: 255 / pixel_dist_scale, pixel_dist_scale = 32.
const float SDF_TEXELS_PER_UNIT = 255.0 / 32.0;

void main() {
    float dist = texture(sampler2D(tex, smp), v_uv).r;

    // Signed distance from the edge, in ATLAS TEXELS. Positive inside.
    float d_texels = (dist - SDF_ONEDGE) * SDF_TEXELS_PER_UNIT;

    // How many texels one screen pixel spans here. fwidth of the UV, which is
    // linear across the primitive -- unlike the sampled distance, whose
    // derivative is undefined in any quad straddling an edge.
    vec2 atlas_size = vec2(textureSize(sampler2D(tex, smp), 0));
    vec2 uv_texels = v_uv * atlas_size;
    float texels_per_pixel = max(length(fwidth(uv_texels)), 1e-4);

    // Distance in screen pixels, turned into coverage across a one-pixel band.
    float d_pixels = d_texels / texels_per_pixel;
    float alpha = clamp(d_pixels + 0.5, 0.0, 1.0);

    // Tint from the per-vertex colour; coverage from the field.
    frag_color = vec4(v_color.rgb, v_color.a * alpha);
}
@end

@program text_sdf vs fs
