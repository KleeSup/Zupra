//------------------------------------------------------------------------------
//  shaders/material_surface.glsl.inc
//
//  BINDING CONTRACT for custom mesh shaders.
//
//  The renderer binds the same resources for every mesh material, so a custom
//  shader has to declare them in the same places or it won't receive them. Any
//  shader that @includes this file automatically matches.
//
//  Why an include rather than "just declare these yourself": sokol-shdc strips
//  bindings a shader never references, so a hand-written shader that declares
//  base_color_map but ignores it ends up with NO base_color_map in the generated
//  slot list — and then the renderer's bind lands on the wrong index. The
//  materialSample() helper below touches every texture, which keeps them all
//  live regardless of what your shader actually uses.
//
//  Slots (must match mesh.zig / deferred.zig bindings):
//      view 0 = normal_map        view 1 = base_color_map
//      view 2 = metallic_roughness_map
//      view 3 = occlusion_map     view 4 = emissive_map
//      sampler 0 = smp_material
//      UB 0 = vs_params (engine)  UB 1 = fs_params (yours)
//      UB 2 = uv_params (engine, optional: omit if you don't need UV transforms)
//------------------------------------------------------------------------------

@block material_textures
layout(binding=0) uniform texture2D base_color_map;          // MapSlot.base_color
layout(binding=1) uniform texture2D normal_map;              // MapSlot.normal
layout(binding=2) uniform texture2D metallic_roughness_map;  // MapSlot.metallic_roughness
layout(binding=3) uniform texture2D occlusion_map;           // MapSlot.occlusion
layout(binding=4) uniform texture2D emissive_map;            // MapSlot.emissive
layout(binding=0) uniform sampler smp_material;

/// Everything the standard material provides at one UV. Call it even if you only
/// want one field, the compiler will fold away what you don't use, but the bindings
/// stay declared, which is the point.
struct SurfaceSample {
    vec4 base_color;
    vec3 normal_ts;   // tangent-space, already unpacked from [0,1] to [-1,1]
    float metallic;
    float roughness;
    float occlusion;
    vec3 emissive;
};

SurfaceSample materialSample(vec2 uv) {
    SurfaceSample s;
    s.base_color = texture(sampler2D(base_color_map, smp_material), uv);
    s.normal_ts = texture(sampler2D(normal_map, smp_material), uv).rgb * 2.0 - 1.0;
    vec3 mr = texture(sampler2D(metallic_roughness_map, smp_material), uv).rgb;
    s.metallic = mr.b;   // glTF packs metallic in B
    s.roughness = mr.g;  // and roughness in G
    s.occlusion = texture(sampler2D(occlusion_map, smp_material), uv).r;
    s.emissive = texture(sampler2D(emissive_map, smp_material), uv).rgb;
    return s;
}
@end

@block material_vs_params
layout(binding=0) uniform vs_params {
    mat4 model;
    mat4 view_proj;
    vec4 uv_scale;
};
@end
