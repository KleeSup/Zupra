//------------------------------------------------------------------------------
//  shaders/bloom_composite.glsl
//
//  Combines the finished bloom with the original HDR scene.
//
//  Runs BEFORE tonemap, on linear light. Bloom is a lens/sensor phenomenon:
//  light scatters on its way to the sensor, so it has to be added while values
//  are still physical. Adding it after tonemap would apply display-referred
//  arithmetic to something that happened optically, and highlights would clip
//  instead of blooming.
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
layout(binding=0) uniform composite_params {
    vec4 params; // x = intensity, yzw unused
};

layout(binding=0) uniform texture2D tex_scene;
layout(binding=1) uniform texture2D tex_bloom;
layout(binding=0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec3 scene = textureLod(sampler2D(tex_scene, smp), v_uv, 0.0).rgb;
    vec3 bloom = textureLod(sampler2D(tex_bloom, smp), v_uv, 0.0).rgb;
    frag_color = vec4(scene + bloom * params.x, 1.0);
}
@end

@program bloom_composite vs fs
