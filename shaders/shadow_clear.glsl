//------------------------------------------------------------------------------
//  shaders/shadow_clear.glsl
//
//  Per-tile shadow-atlas depth clear.  A cached atlas cannot use a whole
//  attachment clear: that would erase valid persistent tiles.  This triangle
//  is clipped by the caller's viewport and scissor, and writes far depth (1.0)
//  only into the tile about to be redrawn.
//------------------------------------------------------------------------------

@vs vs
in vec2 pos;
in vec2 uv;

void main() {
    // Keep uv live so this shader has the standard fullscreen vertex layout.
    float keep = uv.x * 0.0;
    gl_Position = vec4(pos.x + keep, pos.y, 1.0, 1.0);
}
@end

@fs fs
out vec4 frag_color;

void main() {
    // The pass has no colour attachments; this output only keeps the fragment
    // stage valid on backends that require one.  Depth is written by pipeline
    // state with compare ALWAYS.
    frag_color = vec4(1.0);
}
@end

@program shadow_clear vs fs
