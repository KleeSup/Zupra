// Current + previous skin palettes for TAA object/bone velocity.

@block skinning_velocity
@image_sample_type joint_palette unfilterable_float
layout(binding=13) uniform texture2D joint_palette;
@image_sample_type previous_joint_palette unfilterable_float
layout(binding=14) uniform texture2D previous_joint_palette;
@sampler_type smp_joint_palette nonfiltering
layout(binding=4) uniform sampler smp_joint_palette;

const uint ZUPRA_SKIN_MATRICES_PER_ROW = 256u;

mat4 zupraCurrentJointMatrix(uint joint) {
    uint row = joint / ZUPRA_SKIN_MATRICES_PER_ROW;
    uint column = (joint % ZUPRA_SKIN_MATRICES_PER_ROW) * 4u;
    return mat4(
        texelFetch(sampler2D(joint_palette, smp_joint_palette), ivec2(int(column + 0u), int(row)), 0),
        texelFetch(sampler2D(joint_palette, smp_joint_palette), ivec2(int(column + 1u), int(row)), 0),
        texelFetch(sampler2D(joint_palette, smp_joint_palette), ivec2(int(column + 2u), int(row)), 0),
        texelFetch(sampler2D(joint_palette, smp_joint_palette), ivec2(int(column + 3u), int(row)), 0)
    );
}

mat4 zupraPreviousJointMatrix(uint joint) {
    uint row = joint / ZUPRA_SKIN_MATRICES_PER_ROW;
    uint column = (joint % ZUPRA_SKIN_MATRICES_PER_ROW) * 4u;
    return mat4(
        texelFetch(sampler2D(previous_joint_palette, smp_joint_palette), ivec2(int(column + 0u), int(row)), 0),
        texelFetch(sampler2D(previous_joint_palette, smp_joint_palette), ivec2(int(column + 1u), int(row)), 0),
        texelFetch(sampler2D(previous_joint_palette, smp_joint_palette), ivec2(int(column + 2u), int(row)), 0),
        texelFetch(sampler2D(previous_joint_palette, smp_joint_palette), ivec2(int(column + 3u), int(row)), 0)
    );
}

mat4 zupraCurrentSkinMatrix(uvec4 joints, vec4 weights) {
    return zupraCurrentJointMatrix(joints.x) * weights.x +
           zupraCurrentJointMatrix(joints.y) * weights.y +
           zupraCurrentJointMatrix(joints.z) * weights.z +
           zupraCurrentJointMatrix(joints.w) * weights.w;
}

mat4 zupraPreviousSkinMatrix(uvec4 joints, vec4 weights) {
    return zupraPreviousJointMatrix(joints.x) * weights.x +
           zupraPreviousJointMatrix(joints.y) * weights.y +
           zupraPreviousJointMatrix(joints.z) * weights.z +
           zupraPreviousJointMatrix(joints.w) * weights.w;
}
@end
