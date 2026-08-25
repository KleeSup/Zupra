//------------------------------------------------------------------------------
//  shaders/depth_prepass.glsl
//
//  Depth + world-normal prepass for the forward path.
//
//  MUST STAY BIT-IDENTICAL TO mesh.glsl's VERTEX POSITION MATH. The prepass lays
//  down the depth that the shading pass then tests against with LESS_EQUAL, so
//  if the two compute gl_Position differently, even by one ulp, fragments get
//  rejected wherever the prepass rounds nearer and the object is punched full of
//  holes that reshuffle as the camera moves.
//
//  That rules out the obvious shortcut of premultiplying model * view_proj on
//  the CPU and shipping one mvp matrix: the same result in exact arithmetic, a
//  different one in floats. Hence two separate mat4s and the same two
//  multiplies, in the same order, as mesh.glsl. Any change to how mesh.glsl
//  derives gl_Position has to be mirrored here.
//
//  IT ALSO WRITES WORLD NORMALS. Screen-space AO needs depth and normals as
//  textures, and the forward path has no G-buffer to supply them. Emitting the
//  normal here costs one attachment and gives forward exactly the AO input that
//  deferred has. Reconstructing normals from depth derivatives would be free,
//  but yields the triangle's face normal, so smooth surfaces come out faceted.
//
//  Output matches the G-buffer's convention -- world normal in rgb, alpha as a
//  coverage mask -- so the AO shader is identical for both paths.
//
//  Alpha-masked materials sample their base-colour alpha here too. Without that
//  discard, foliage and grates lay down solid prepass depth/normals while the
//  colour pass punches holes through them, corrupting both early-z and AO.
//
//  Reuses the .mesh vertex layout (pos/normal/uv/tangent/uv1). Tangent is kept
//  referenced even though this pass does not use it, so shdc does NOT strip it:
//  the pipeline binds the full .mesh layout, and a generated vertex-input
//  signature missing attributes would mismatch it and fail pipeline creation.
//------------------------------------------------------------------------------

@include pbr_lib.glsl.inc
@include material_alpha.glsl.inc

@vs vs
layout(binding=0) uniform vs_params {
    mat4 model;
    mat4 view_proj;
    vec4 uv_scale;
};

in vec3 pos;
in vec3 normal;
in vec2 uv;
in vec4 tangent;
in vec2 uv1;

out vec3 v_normal;
out vec2 v_uv;
out vec2 v_uv1;

void main() {
    vec4 world = model * vec4(pos, 1.0);
    gl_Position = view_proj * world;

    // Rotate the normal into world space. mat3() of a mat4 drops translation,
    // which is what a direction wants. This is the same approximation mesh.glsl
    // makes (a non-uniform scale would want the inverse transpose), and it has
    // to stay the same, or AO and shading would disagree about which way a
    // surface faces.
    v_normal = mat3(model) * normal;
    v_uv = uv * uv_scale.xy;
    v_uv1 = uv1 * uv_scale.xy;

    // Keep the remaining unused attributes live. Applied AFTER the transform and
    // as an addition of exact zero, so the position arithmetic above is
    // untouched; perturbing `pos` before the multiply would break the bit
    // equality with mesh.glsl that this pass depends on.
    gl_Position.x += tangent.x * 0.0;
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

in vec3 v_normal;
in vec2 v_uv;
in vec2 v_uv1;
out vec4 frag_color;

@include_block uv_transform
@include_block material_alpha

void main() {
    // Most prepass draws are opaque. Keep their original position/normal-only
    // cost and sample the base-colour map only for a mask, where coverage is
    // genuinely needed to decide depth.
    if (alpha_params.y > 0.5) {
        vec2 uv_bc = mapUv(UV_BASE_COLOR, v_uv, v_uv1);
        vec4 base_sample = texture(sampler2D(base_color_map, smp_material), uv_bc);
        discardMasked(materialAlpha(base_color, base_sample), alpha_params);
    }

    // alpha = 1 marks "geometry here". The AO pass reads it exactly as it reads
    // the G-buffer's normal alpha, so background pixels are recognised the same
    // way in both paths.
    frag_color = vec4(normalize(v_normal), 1.0);
}
@end

@program depth_prepass vs fs
