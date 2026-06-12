//------------------------------------------------------------------------------
//  shaders/tonemap.glsl
//
//  Present pass. Samples the linear-HDR scene-color target and maps it to a
//  displayable LDR image: exposure, then Reinhard tonemap, then gamma. This is
//  the LAST step of the frame, it's separated from the lighting pass so that
//  lighting and transparent compositing both happen in linear HDR, and only the
//  final image is tonemapped (correct order; also where bloom would sit later).
//
//  tonemap_params (binding 0): params.x = exposure
//------------------------------------------------------------------------------

@vs vs
in vec2 pos;
in vec2 uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
    v_uv = uv;
}
@end

@fs fs
layout(binding=0) uniform texture2D tex_hdr;
layout(binding=0) uniform sampler smp;
layout(binding=0) uniform tonemap_params {
    vec4 params; // x = exposure
};

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec3 hdr = texture(sampler2D(tex_hdr, smp), v_uv).rgb;
    hdr *= params.x;                       // exposure
    vec3 color = hdr / (hdr + vec3(1.0));  // Reinhard tonemap
    color = pow(color, vec3(1.0 / 2.2));   // gamma
    frag_color = vec4(color, 1.0);
}
@end

@program tonemap vs fs
