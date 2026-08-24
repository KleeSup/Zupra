//------------------------------------------------------------------------------
//  shaders/xegtao_main.glsl
//
//  GTAO main pass. Ported to GLSL from the reference implementation:
//
//      XeGTAO, Copyright (C) 2016-2021, Intel Corporation, SPDX MIT
//      Filip Strugar, Steve McCalla — https://github.com/GameTechDev/XeGTAO
//      based on Jimenez et al. 2016, "Practical Real-Time Strategies for
//      Accurate Indirect Occlusion"
//
//  Differences from the reference are forced by this renderer, not chosen:
//  fragment shaders instead of compute, so the depth pyramid is built by
//  repeated passes rather than one dispatch with groupshared memory; and an
//  explicit origin flip, since the reference targets D3D where texture origin is
//  always top-left.
//
//  The algorithm itself follows the reference closely, including several details
//  that turn out to matter far more than they look. They are marked below.
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
layout(binding=0) uniform gtao_params {
    // World -> view, for rotating the G-buffer's world-space normal.
    mat4 view;
    // xy = tan(half fov) per axis, zw = its negation for the y flip. Together
    // these turn a screen uv straight into a view ray, matching the reference's
    // NDCToViewMul / NDCToViewAdd.
    vec4 ndc_to_view_mul;
    vec4 ndc_to_view_add;
    // xy = 1 / viewport size, zw = viewport size in pixels.
    vec4 viewport;
    // x = effect radius (world), y = falloff range as a fraction of radius,
    // z = sample distribution power, w = thin occluder compensation.
    vec4 radius_params;
    // x = slice count, y = steps per slice, z = final value power,
    // w = depth mip sampling offset.
    vec4 counts;
    // x = frame index, y = 1 when temporal jitter is on, zw unused.
    vec4 temporal;
    // x = highest live pyramid mip, yzw unused. Not a constant: this build
    // creates fewer than five levels at small window sizes.
    vec4 pyramid;
};

// The pyramid is split across two images by level parity, because sokol cannot
// bind a texture from the image it is currently writing.
layout(binding=0) uniform texture2D tex_depth_even;
layout(binding=2) uniform texture2D tex_depth_odd;
layout(binding=1) uniform texture2D tex_normal;
layout(binding=0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

const float XE_PI = 3.1415926535897932384626433832795;
const float XE_PI_HALF = 1.5707963267948966192313216916398;

// If an offset is under about a pixel, push it out. A tap on the pixel itself
// carries no information.
const float PIXEL_TOO_CLOSE_THRESHOLD = 1.3;

// Input [-1,1], output [0,PI]. From the reference, after Lagarde.
float fastACos(float inX) {
    float x = abs(inX);
    float res = -0.156583 * x + XE_PI_HALF;
    res *= sqrt(1.0 - x);
    return (inX >= 0.0) ? res : XE_PI - res;
}

float sampleDepthMip(vec2 uv, float mip) {
    // `textureLod` with the reference point-mip sampler picks the nearest
    // level for a fractional LOD. Round before choosing an image so the
    // parity and the level always agree (flooring kept noisy mip 0 active for
    // too much of the search radius).
    float m = floor(mip + 0.5);
    if (mod(m, 2.0) < 0.5) return textureLod(sampler2D(tex_depth_even, smp), uv, m).r;
    return textureLod(sampler2D(tex_depth_odd, smp), uv, m).r;
}

// Screen uv plus view depth -> view-space position.
//
// Takes the UV directly rather than converting to NDC first, exactly as the
// reference does. ndc_to_view_mul carries the y flip in its sign, so there is
// one conversion here and no separate flip anywhere else. Applying an origin
// flip on top of this is what mirrored the pyramid in an earlier attempt.
vec3 computeViewspacePosition(vec2 screen_pos, float viewspace_depth) {
    vec3 ret;
    ret.xy = (ndc_to_view_mul.xy * screen_pos + ndc_to_view_add.xy) * viewspace_depth;
    ret.z = viewspace_depth;
    return ret;
}

// The reference XeGTAO ordering. A Hilbert curve gives neighbouring pixels
// well-spaced values in the R2 sequence instead of placing an entire diagonal
// on the same noise value. The old `3*x + 7*y` index repeated exactly along
// (7, -3) diagonals, which is the screen-space hatching visible under TAA.
uint hilbertIndex(uvec2 pos) {
    const uint width = 64u;
    uint index = 0u;
    for (uint level = width / 2u; level > 0u; level /= 2u) {
        uint region_x = (pos.x & level) > 0u ? 1u : 0u;
        uint region_y = (pos.y & level) > 0u ? 1u : 0u;
        index += level * level * ((3u * region_x) ^ region_y);
        if (region_y == 0u) {
            if (region_x == 1u) {
                pos = (width - 1u) - pos;
            }
            pos = pos.yx;
        }
    }
    return index;
}

// XeGTAO's Hilbert-driven R2 sequence. `temporal_index` is zero without TAA
// and cycles through the reference's 64-frame sequence when TAA is active.
vec2 spatialDirectionalNoise(uvec2 pix_coord, uint temporal_index) {
    uint index = hilbertIndex(pix_coord);
    index += 288u * (temporal_index % 64u);
    return fract(0.5 + float(index) * vec2(
        0.75487766624669276005,
        0.56984029099805326591
    ));
}

void main() {
    vec4 nrm = textureLod(sampler2D(tex_normal, smp), v_uv, 0.0);
    if (nrm.a < 0.5) {
        frag_color = vec4(1.0);
        return;
    }

    vec2 normalized_screen_pos = v_uv;
    float viewspace_z = sampleDepthMip(v_uv, 0.0);

    // Rows arrive as columns, so view * v is the row-vector transform. w = 0
    // because a normal is a direction and translation must not apply.
    vec3 viewspace_normal = normalize((view * vec4(normalize(nrm.xyz), 0.0)).xyz);

    // Move the centre pixel very slightly toward the camera.
    //
    // Small, and load bearing. Without it, a sample on the SAME flat surface can
    // reconstruct fractionally in front of the centre point, which registers as
    // a horizon above the tangent plane and darkens the surface. The error grows
    // with grazing angle, so a flat floor picks up a gradient that deepens
    // toward the bottom of the screen. 0.99999 is the reference value for a
    // 32-bit depth buffer.
    viewspace_z *= 0.99999;

    vec3 pix_center_pos = computeViewspacePosition(normalized_screen_pos, viewspace_z);
    vec3 view_vec = normalize(-pix_center_pos);

    float effect_radius = radius_params.x;
    float falloff_range_frac = radius_params.y;
    float sample_distribution_power = radius_params.z;
    float thin_occluder_compensation = radius_params.w;

    float falloff_range = falloff_range_frac * effect_radius;
    float falloff_from = effect_radius * (1.0 - falloff_range_frac);
    float falloff_mul = -1.0 / falloff_range;
    float falloff_add = falloff_from / falloff_range + 1.0;

    float slice_count = counts.x;
    float steps_per_slice = counts.y;

    float visibility = 0.0;

    // Approximate view-space size of one pixel at this depth, which converts the
    // world radius into a screen radius.
    float pixel_dir_rb_viewspace_size_at_center_z = abs(viewspace_z * ndc_to_view_mul.x * viewport.x);
    float screenspace_radius = effect_radius / max(pixel_dir_rb_viewspace_size_at_center_z, 1e-6);

    // Fade out where the radius spans too few pixels to carry information.
    //
    // NOT a clamp. Clamping the radius, which earlier attempts did, silently
    // changes the volume being measured depending on distance, so the same
    // geometry gives different answers from different viewpoints. Fading admits
    // there is nothing to measure instead.
    visibility += clamp((10.0 - screenspace_radius) / 100.0, 0.0, 1.0) * 0.5;

    float min_s = PIXEL_TOO_CLOSE_THRESHOLD / screenspace_radius;

    vec2 local_noise = spatialDirectionalNoise(
        uvec2(gl_FragCoord.xy),
        (temporal.y > 0.5) ? uint(temporal.x) : 0u
    );
    float noise_slice = local_noise.x;
    float noise_sample = local_noise.y;

    for (float slice = 0.0; slice < slice_count; slice += 1.0) {
        float slice_k = (slice + noise_slice) / slice_count;
        float phi = slice_k * XE_PI;
        float cos_phi = cos(phi);
        float sin_phi = sin(phi);

        // NOTE THE SIGNS. omega is a screen-space offset, where +y runs DOWN;
        // direction_vec is the same direction in view space, where +y runs UP.
        // They differ by the sign of sin(phi), and using one sign for both, as an
        // earlier attempt did, mirrors every slice against the surface it is
        // measuring.
        vec2 omega = vec2(cos_phi, -sin_phi) * screenspace_radius;
        vec3 direction_vec = vec3(cos_phi, sin_phi, 0.0);

        vec3 ortho_direction_vec = direction_vec - (dot(direction_vec, view_vec) * view_vec);
        vec3 axis_vec = normalize(cross(ortho_direction_vec, view_vec));

        vec3 projected_normal_vec = viewspace_normal - axis_vec * dot(viewspace_normal, axis_vec);
        float sign_norm = sign(dot(ortho_direction_vec, projected_normal_vec));
        float projected_normal_vec_length = length(projected_normal_vec);

        // saturate, not clamp to [-1,1]. A projected normal facing away from the
        // viewer is degenerate here, and letting the cosine go negative sends
        // the arc integral somewhere meaningless.
        float cos_norm = clamp(dot(projected_normal_vec, view_vec) / max(projected_normal_vec_length, 1e-6), 0.0, 1.0);
        float n = sign_norm * fastACos(cos_norm);

        // The unoccluded horizon is the TANGENT PLANE, not -1. Minus one is 180
        // degrees from the view vector, which on a tilted surface lies below the
        // true horizon, so the search would climb up from an artificial floor and
        // open surfaces would come out darkened.
        float low_horizon_cos0 = cos(n + XE_PI_HALF);
        float low_horizon_cos1 = cos(n - XE_PI_HALF);

        float horizon_cos0 = low_horizon_cos0;
        float horizon_cos1 = low_horizon_cos1;

        for (float step = 0.0; step < steps_per_slice; step += 1.0) {
            // R1 sequence offset per step.
            float step_base_noise = (slice + step * steps_per_slice) * 0.6180339887498948482;
            float step_noise = fract(noise_sample + step_base_noise);

            float s = (step + step_noise) / steps_per_slice;
            // Dense near the pixel, sparse further out: occlusion detail lives
            // close to the surface.
            s = pow(s, sample_distribution_power);
            s += min_s;

            vec2 sample_offset = s * omega;
            float sample_offset_length = length(sample_offset);

            // Read a coarser pyramid level the further out the sample is. This
            // is what lets a fixed, small tap count cover the whole radius from
            // any distance, and it is the piece that makes the screen radius
            // safe to leave unclamped.
            float mip_level = clamp(log2(sample_offset_length) - counts.w, 0.0, pyramid.x);

            // Snap to pixel centres, so the direction maths matches the texel
            // actually sampled.
            sample_offset = round(sample_offset) * viewport.xy;

            vec2 sample_screen_pos0 = normalized_screen_pos + sample_offset;
            vec2 sample_screen_pos1 = normalized_screen_pos - sample_offset;

            float sz0 = sampleDepthMip(sample_screen_pos0, mip_level);
            float sz1 = sampleDepthMip(sample_screen_pos1, mip_level);

            vec3 sample_pos0 = computeViewspacePosition(sample_screen_pos0, sz0);
            vec3 sample_pos1 = computeViewspacePosition(sample_screen_pos1, sz1);

            vec3 sample_delta0 = sample_pos0 - pix_center_pos;
            vec3 sample_delta1 = sample_pos1 - pix_center_pos;
            float sample_dist0 = length(sample_delta0);
            float sample_dist1 = length(sample_delta1);

            vec3 sample_horizon_vec0 = sample_delta0 / max(sample_dist0, 1e-6);
            vec3 sample_horizon_vec1 = sample_delta1 / max(sample_dist1, 1e-6);

            // THE THIN OCCLUDER HEURISTIC, and it is not a separate depth test.
            // Stretching the delta's z before measuring its length makes a sample
            // that sits far behind the surface read as further away, so the
            // ordinary falloff discards it. That covers geometry seen past an
            // edge or through a gap, which an earlier attempt handled with an ad
            // hoc depth comparison that never worked properly.
            float falloff_base0 = length(vec3(sample_delta0.x, sample_delta0.y,
                sample_delta0.z * (1.0 + thin_occluder_compensation)));
            float falloff_base1 = length(vec3(sample_delta1.x, sample_delta1.y,
                sample_delta1.z * (1.0 + thin_occluder_compensation)));

            float weight0 = clamp(falloff_base0 * falloff_mul + falloff_add, 0.0, 1.0);
            float weight1 = clamp(falloff_base1 * falloff_mul + falloff_add, 0.0, 1.0);

            float shc0 = dot(sample_horizon_vec0, view_vec);
            float shc1 = dot(sample_horizon_vec1, view_vec);

            // Discarded samples fade back to the LOW HORIZON, not to -1.
            shc0 = mix(low_horizon_cos0, shc0, weight0);
            shc1 = mix(low_horizon_cos1, shc1, weight1);

            horizon_cos0 = max(horizon_cos0, shc0);
            horizon_cos1 = max(horizon_cos1, shc1);
        }

        // The reference's fudge for slight over-darkening on high slopes.
        projected_normal_vec_length = mix(projected_normal_vec_length, 1.0, 0.05);

        float h0 = -fastACos(horizon_cos1);
        float h1 = fastACos(horizon_cos0);

        // No clamping of h0/h1 into the hemisphere. The reference leaves it out
        // by default, and adding it, as an earlier attempt did, changes the
        // integral's result on tilted surfaces.

        float iarc0 = (cos_norm + 2.0 * h0 * sin(n) - cos(2.0 * h0 - n)) / 4.0;
        float iarc1 = (cos_norm + 2.0 * h1 * sin(n) - cos(2.0 * h1 - n)) / 4.0;

        visibility += projected_normal_vec_length * (iarc0 + iarc1);
    }

    visibility /= slice_count;
    visibility = pow(max(visibility, 0.0), counts.z);
    // Never fully occluded: a visible pixel receives some light in any real
    // scene, and zero reads as a hole rather than a shadow.
    visibility = max(0.03, visibility);

    frag_color = vec4(visibility);
}
@end

@program xegtao_main vs fs
