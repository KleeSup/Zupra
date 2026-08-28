// GPU skin palette shared by every built-in skinned vertex stage.
//
// A joint matrix occupies four RGBA32F texels. `mat4(a,b,c,d)` treats its
// arguments as COLUMNS, which deliberately matches the engine's raw row-vector
// Matrix upload convention: the four CPU rows are the GLSL matrix columns.

@block skinning
@image_sample_type joint_palette unfilterable_float
layout(binding=13) uniform texture2D joint_palette;
@sampler_type smp_joint_palette nonfiltering
layout(binding=4) uniform sampler smp_joint_palette;

const uint ZUPRA_SKIN_MATRICES_PER_ROW = 256u;

mat4 zupraJointMatrix(uint joint) {
    uint row = joint / ZUPRA_SKIN_MATRICES_PER_ROW;
    uint column = (joint % ZUPRA_SKIN_MATRICES_PER_ROW) * 4u;
    return mat4(
        texelFetch(sampler2D(joint_palette, smp_joint_palette), ivec2(int(column + 0u), int(row)), 0),
        texelFetch(sampler2D(joint_palette, smp_joint_palette), ivec2(int(column + 1u), int(row)), 0),
        texelFetch(sampler2D(joint_palette, smp_joint_palette), ivec2(int(column + 2u), int(row)), 0),
        texelFetch(sampler2D(joint_palette, smp_joint_palette), ivec2(int(column + 3u), int(row)), 0)
    );
}

mat4 zupraSkinMatrix(uvec4 joints, vec4 weights) {
    return zupraJointMatrix(joints.x) * weights.x +
           zupraJointMatrix(joints.y) * weights.y +
           zupraJointMatrix(joints.z) * weights.z +
           zupraJointMatrix(joints.w) * weights.w;
}
@end
