//------------------------------------------------------------------------------
//  shaders/shadow_depth_instanced.glsl
//
//  Instanced position-only depth, for the shadow atlas passes.
//
//  Same output as shadow_depth.glsl, but the model matrix arrives as four
//  per-instance vertex attributes instead of a uniform, so every caster sharing
//  a mesh collapses into one draw. The shadow passes are where this pays: the
//  same geometry is drawn into every cascade and every cube face, so the draw
//  count is queue length times view count before batching.
//
//  Position-only is what makes the batching effective. Material has no bearing
//  on depth, so the batch key is the mesh alone -- twelve differently coloured
//  spheres are one draw here, where the shading pass would need twelve.
//
//  Unlike depth_prepass.glsl this does NOT have to match mesh.glsl bit for bit.
//  Nothing compares this depth against another pass's; the shadow test compares
//  it against itself, one frame's atlas against that same frame's fragments.
//
//  Vertex layout = VertexLayout.mesh_instanced: buffer 0 is the usual Vertex3D
//  stream, buffer 1 steps per instance and holds the model matrix as four rows.
//------------------------------------------------------------------------------

@vs vs
layout(binding=0) uniform vs_params {
    mat4 view_proj;
};

in vec3 pos;
in vec3 normal;
in vec2 uv;
in vec4 tangent;
in vec2 uv1;

// Per-instance model matrix, four ROWS (the CPU side is row-major). mat4()
// consumes them as columns, producing the transpose -- which is what the
// M * v multiply below wants, exactly as in the uniform case where std140
// hands the shader a column-major view of the same bytes.
in vec4 i_row0;
in vec4 i_row1;
in vec4 i_row2;
in vec4 i_row3;

void main() {
    mat4 model = mat4(i_row0, i_row1, i_row2, i_row3);
    vec4 world = model * vec4(pos, 1.0);
    gl_Position = view_proj * world;

    // Keep the unused per-vertex attributes referenced so shdc doesn't strip
    // them and leave a vertex-input signature that no longer matches the .mesh
    // half of the layout. Exact zero, applied after the transform.
    gl_Position.x += (normal.x + uv.x + tangent.x + uv1.x) * 0.0;
}
@end

@fs fs
out vec4 frag_color;
void main() {
    // Depth-only in practice; the pass has no colour attachment. An explicit
    // output keeps the fragment stage valid on backends that reject an empty
    // pixel shader.
    frag_color = vec4(1.0);
}
@end

@program shadow_depth_instanced vs fs
