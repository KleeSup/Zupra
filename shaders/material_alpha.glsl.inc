//------------------------------------------------------------------------------
//  shaders/material_alpha.glsl.inc
//
//  Shared glTF base-colour alpha semantics for every built-in material pass.
//  The RGB channels are handled by the individual shading paths because they
//  have different colour spaces and lighting, but alpha is always the straight
//  product of baseColorFactor.a and baseColorTexture.a. Alpha masking must use
//  that product, not texture alpha alone, or material factors disagree between
//  the colour, depth, velocity and shadow passes.
//------------------------------------------------------------------------------

@block material_alpha
float materialAlpha(vec4 base_color_factor, vec4 base_color_sample) {
    return base_color_factor.a * base_color_sample.a;
}

/// alpha_params.x = alphaCutoff, alpha_params.y = alpha-mode-is-MASK.
/// The mode flag is separate because 0 is a legal glTF cutoff.
void discardMasked(float alpha, vec4 alpha_params) {
    if (alpha_params.y > 0.5 && alpha < alpha_params.x) discard;
}
@end
