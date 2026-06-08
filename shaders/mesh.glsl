//------------------------------------------------------------------------------
//  shaders/mesh.glsl
//
//  Forward lit mesh shader. Single directional light using only diffuse and ambient.
//
//  THIS FILE IS A CONTRACT. The Mesh draw path builds its pipeline against
//  these inputs; a custom mesh shader is drop-in if it preserves them:
//
//    Vertex inputs (= VertexLayout.mesh in pipeline.zig):
//      location 0  pos     vec3  (Vertex3D.pos)
//      location 1  normal  vec3  (Vertex3D.normal)
//      location 2  uv      vec2  (Vertex3D.uv)
//
//    Vertex uniform block `vs_params` (binding 0):
//      mat4 model      -- object -> world (per draw)
//      mat4 view_proj  -- world  -> clip  (camera.viewProjection())
//    Kept SEPARATE on purpose: lighting needs the WORLD-space position/normal,
//    not just the combined mvp.
//
//    Fragment uniform block `fs_params` (binding 1):
//      vec4 base_color   -- albedo tint (rgb) + alpha
//      vec4 light_dir    -- xyz: normalized direction FROM surface TO the light
//      vec4 light_color  -- xyz
//      vec4 ambient      -- xyz
//
//  MATRIX RULE: zmath is row-major, GLSL reads
//  column-major, so upload model / view_proj DIRECTLY (no transpose). Then
//  `view_proj * model * vec4(pos,1)` reproduces the zmath model-then-viewproj
//  transform, and `mat3(model) * normal` rotates the normal correctly.
//
//  NORMALS: mat3(model) is correct for rotation + uniform scale (the common
//  case). Non-uniform scale needs a proper inverse-transpose normal matrix —
//  a later addition when materials land.
//------------------------------------------------------------------------------

@vs vs
layout(binding=0) uniform vs_params {
    mat4 model;
    mat4 view_proj;
};

in vec3 pos;
in vec3 normal;
in vec2 uv;

out vec3 v_world_normal;
out vec2 v_uv;

void main() {
    vec4 world = model * vec4(pos, 1.0);
    gl_Position = view_proj * world;
    v_world_normal = mat3(model) * normal;
    v_uv = uv;
}
@end

@fs fs
layout(binding=1) uniform fs_params {
    vec4 base_color;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient;
};

in vec3 v_world_normal;
in vec2 v_uv;

out vec4 frag_color;

void main() {
    vec3 N = normalize(v_world_normal);
    vec3 L = normalize(light_dir.xyz);
    float ndl = max(dot(N, L), 0.0);

    vec3 lit = base_color.rgb * (ambient.xyz + light_color.xyz * ndl);
    frag_color = vec4(lit, base_color.a);
}
@end

@program mesh vs fs
