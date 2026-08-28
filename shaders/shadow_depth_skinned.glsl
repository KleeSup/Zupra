@include skinning.glsl.inc

@vs vs
layout(binding=0) uniform vs_params {
    mat4 mvp;
};

in vec3 pos;
in vec3 normal;
in vec2 uv;
in vec4 tangent;
in vec2 uv1;
in uvec4 joints;
in vec4 weights;

@include_block skinning

void main() {
    mat4 skin = zupraSkinMatrix(joints, weights);
    float keep = (normal.x + uv.x + tangent.x + uv1.x) * 0.0;
    gl_Position = mvp * (skin * vec4(pos.x + keep, pos.y, pos.z, 1.0));
}
@end

@fs fs
out vec4 frag_color;
void main() {
    frag_color = vec4(1.0);
}
@end

@program shadow_depth_skinned vs fs
