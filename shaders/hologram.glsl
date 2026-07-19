//------------------------------------------------------------------------------
//  shaders/hologram.glsl
//
//  EXAMPLE CUSTOM MESH SHADER — animated fresnel hologram with scanlines.
//
//  Demonstrates the per-material custom shader path: this replaces the built-in
//  PBR/lambert/unlit shader for whichever materials carry it, so you can give
//  ONE model a bespoke look without touching the rest of the scene.
//
//  It does its own shading (no lights, no IBL), so it belongs on the FORWARD
//  path — leave Material.shader_writes_gbuffer = false and the scene routes it
//  there automatically, even in deferred mode. A custom shader that wanted to
//  participate in deferred lighting would instead write the four G-buffer
//  targets; see gbuffer.glsl for that layout.
//
//    fs_params (UB 1):
//      tint.rgb   hologram colour       tint.w   overall opacity
//      params.x   fresnel power (2 broad rim .. 8 thin rim)
//      params.y   scanline frequency in world units
//      params.z   scanline speed
//      params.w   time (seconds), fed each frame
//------------------------------------------------------------------------------

@include material_surface.glsl.inc

@vs vs
@include_block material_vs_params

in vec3 pos;
in vec3 normal;
in vec2 uv;
in vec4 tangent;
in vec2 uv1;

out vec3 v_world_pos;
out vec3 v_world_normal;
out vec2 v_uv;

void main() {
    vec4 wp = model * vec4(pos, 1.0);
    v_world_pos = wp.xyz;
    // Normal matrix: fine for uniform scale. Non-uniform scale needs the
    // inverse-transpose uploaded separately.
    v_world_normal = normalize((model * vec4(normal, 0.0)).xyz);
    v_uv = uv * uv_scale.xy;
    gl_Position = view_proj * wp;

    // Keep the unused inputs live so the vertex layout still matches .mesh —
    // shdc would otherwise drop the attributes and shift the slot indices.
    v_uv += (tangent.xy + uv1) * 0.0;
}
@end

@fs fs
@include_block material_textures

layout(binding=1) uniform fs_params {
    vec4 tint;   // rgb colour, w opacity
    vec4 params; // fresnel power, scan frequency, scan speed, time
};

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;
out vec4 frag_color;

void main() {
    // The camera position isn't in this shader's uniforms, so derive the view
    // direction from the inverse view-projection implicitly: for a hologram the
    // effect reads fine using the world-space normal against a fixed up-ish
    // view. For a view-correct fresnel, add camera position to fs_params.
    vec3 n = normalize(v_world_normal);
    vec3 view_dir = normalize(vec3(0.0, 0.35, 1.0));

    // Fresnel: surfaces facing away from the viewer glow at the silhouette.
    float facing = 1.0 - abs(dot(n, view_dir));
    float rim = pow(clamp(facing, 0.0, 1.0), max(params.x, 0.5));

    // Scanlines travelling up the model in world space, so they don't swim when
    // the model rotates the way UV-space lines would.
    float scan = 0.5 + 0.5 * sin(v_world_pos.y * params.y - params.z * params.w);
    scan = mix(0.75, 1.0, scan);

    // The material's albedo still contributes, so a textured model keeps its
    // identity under the effect rather than becoming a flat silhouette.
    SurfaceSample s = materialSample(v_uv);
    vec3 albedo = pow(max(s.base_color.rgb, vec3(0.0)), vec3(2.2));

    vec3 col = tint.rgb * (rim * 2.0 + 0.15) * scan + albedo * 0.25;
    float alpha = clamp((rim * 0.85 + 0.15) * tint.w, 0.0, 1.0);

    frag_color = vec4(col, alpha);
}
@end

@program hologram vs fs
