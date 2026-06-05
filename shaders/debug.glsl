//------------------------------------------------------------------------------
//  shaders/debug.glsl
//
//  Debug-draw shader for lines and flat-filled shapes (2D now, 3D later).
//  Vertex layout = VertexLayout.debug (pos: vec3 @0, color0: ubyte4n @1).
//  Same vs_params { mvp } contract as the sprite shader, so the camera's
//  view-projection feeds in identically. No texture, no sampler — just color.
//------------------------------------------------------------------------------

@vs vs
layout(binding=0) uniform vs_params {
    mat4 mvp;
};

in vec3 pos;
in vec4 color0;

out vec4 v_color;

void main() {
    gl_Position = mvp * vec4(pos, 1.0);
    v_color = color0;
}
@end

@fs fs
in vec4 v_color;
out vec4 frag_color;

void main() {
    frag_color = v_color;
}
@end

@program debug vs fs
