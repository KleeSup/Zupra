//------------------------------------------------------------------------------
//  shaders/shadow_depth_alpha.glsl
//
//  Material-aware depth-only shadow caster for glTF alpha MASK materials.
//  Opaque casters stay on shadow_depth_instanced.glsl; this separate path is
//  deliberately used only when base-colour alpha must decide coverage.
//
//  The texture transform and factor/cutoff are the same contract used by the
//  colour, prepass and velocity shaders. A leaf that is visually absent must
//  never still write solid depth into the shadow atlas.
//------------------------------------------------------------------------------

@include pbr_lib.glsl.inc
@include material_alpha.glsl.inc

@vs vs
layout(binding=0) uniform vs_params {
    mat4 mvp;
    vec4 uv_scale;
};

in vec3 pos;
in vec3 normal;
in vec2 uv;
in vec4 tangent;
in vec2 uv1;

out vec2 v_uv;
out vec2 v_uv1;

void main() {
    gl_Position = mvp * vec4(pos, 1.0);
    v_uv = uv * uv_scale.xy;
    v_uv1 = uv1 * uv_scale.xy;

    // The alpha path does not need these attributes, but uses the full .mesh
    // layout. Keep them live so shdc preserves that input signature.
    gl_Position.x += (normal.x + tangent.x) * 0.0;
}
@end

@fs fs
layout(binding=0) uniform texture2D base_color_map;
layout(binding=0) uniform sampler smp_material;

layout(binding=1) uniform fs_params {
    vec4 base_color;
    vec4 alpha_params; // x cutoff, y = 1 for glTF alpha MASK
};

layout(binding=2) uniform uv_params {
    vec4 uv_m[5];
    vec4 uv_aux[5];
};

in vec2 v_uv;
in vec2 v_uv1;
out vec4 frag_color;

@include_block uv_transform
@include_block material_alpha

void main() {
    vec2 uv_bc = mapUv(UV_BASE_COLOR, v_uv, v_uv1);
    vec4 base_sample = texture(sampler2D(base_color_map, smp_material), uv_bc);
    discardMasked(materialAlpha(base_color, base_sample), alpha_params);

    // The atlas pass has no colour attachment. An explicit output keeps every
    // backend's fragment stage valid; sokol discards the colour write.
    frag_color = vec4(1.0);
}
@end

@program shadow_depth_alpha vs fs
