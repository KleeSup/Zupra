//------------------------------------------------------------------------------
//  shaders/taa_resolve.glsl
//
//  Temporal anti-aliasing resolve.
//
//  Each frame renders with the projection nudged by a fraction of a pixel, so
//  consecutive frames sample different points inside every pixel. Accumulating
//  them recovers detail no single frame contains -- which is why TAA both
//  anti-aliases geometry AND averages out per-pixel noise in stochastic effects
//  like GTAO.
//
//  Three things decide whether this looks like anti-aliasing or like smeared
//  mud, and all three are here:
//
//  1. REPROJECTION. Last frame's pixel is elsewhere this frame, so the history
//     is read from where this surface WAS: reconstruct position from depth, push
//     it through the previous view-projection. Exact for static geometry under
//     any camera motion. Objects moving under their own transform are not
//     tracked and rely on (2).
//
//  2. VARIANCE CLIPPING, in YCoCg. History is only trusted where it agrees with
//     what is on screen now. Rather than a min/max box of the 3x3 neighbourhood
//     -- which one outlier pixel can stretch until it accepts almost anything --
//     this builds the neighbourhood's mean and standard deviation and clips to
//     mean +/- gamma*sigma. Tighter, so it catches ghosting the box misses, and
//     more stable, because it moves smoothly with the neighbourhood instead of
//     jumping when the extreme pixel changes. YCoCg because clipping in RGB
//     drags hue as well as brightness, producing coloured fringes on edges.
//
//     CLIPPING, not clamping: the history is moved ALONG THE LINE toward the
//     neighbourhood mean until it enters the box, which preserves its direction
//     in colour space. Per-channel clamping bends the colour toward a box corner
//     and shifts hue.
//
//  3. CATMULL-ROM HISTORY SAMPLING. The reprojected position almost never lands
//     on a texel centre, so the history is resampled every single frame. With
//     bilinear that is a low-pass filter applied hundreds of times over, and it
//     is the entire reason TAA has a reputation for softening the image. A
//     sharpening bicubic kernel very nearly cancels the loss, which is why this
//     costs five taps instead of one.
//------------------------------------------------------------------------------

@vs vs
in vec2 pos;
in vec2 uv;

out vec2 v_uv;
out vec2 v_ndc;

void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
    v_ndc = pos;
    v_uv = uv;
}
@end

@fs fs
layout(binding=0) uniform taa_params {
    // Current inverse view-projection, including the rasterization jitter.
    mat4 inv_view_proj;
    // Previous frame's jittered view-projection.
    mat4 prev_view_proj;
    // xy = 1 / target size, z = blend weight for the current frame,
    // w = 1 when the render target reads top-left first.
    vec4 params;
    // x = 1 to reset history, y = variance clipping gamma,
    // z = history sharpening 0..1, w = 1 when a velocity buffer is bound.
    vec4 flags;
    // x/y convert raw hardware depth to view-space depth via B / (z - A).
    // z is the relative depth tolerance for static-history validation.
    vec4 depth_params;
};

layout(binding=0) uniform texture2D tex_color;
layout(binding=1) uniform texture2D tex_history;
layout(binding=2) uniform texture2D tex_depth;
layout(binding=3) uniform texture2D tex_velocity;
layout(binding=0) uniform sampler smp;
layout(binding=1) uniform sampler smp_point;

in vec2 v_uv;
in vec2 v_ndc;
out vec4 frag_color;

float luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Weight bright samples down before averaging and undo it after, so one very
// bright sample cannot dominate the running mean and flicker for frames.
vec3 tonemapWeight(vec3 c) {
    return c / (1.0 + luma(c));
}
vec3 tonemapUnweight(vec3 c) {
    return c / max(1.0 - luma(c), 1e-4);
}

// YCoCg separates luminance from two chroma axes, so a clip bounded per channel
// bounds brightness and hue independently instead of mixing them.
vec3 rgbToYcocg(vec3 c) {
    return vec3(
        0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
        0.5 * c.r - 0.5 * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b
    );
}
vec3 ycocgToRgb(vec3 c) {
    float t = c.x - c.z;
    return vec3(t + c.y, c.x + c.z, t - c.y);
}

// Move `history` along the line toward `center` until it lies inside the box.
// Preserves its direction in colour space; a per-channel clamp would not.
vec3 clipToBox(vec3 box_min, vec3 box_max, vec3 history) {
    vec3 center = 0.5 * (box_max + box_min);
    vec3 extent = 0.5 * (box_max - box_min) + 1e-5;
    vec3 v = history - center;
    vec3 units = abs(v / extent);
    float worst = max(units.x, max(units.y, units.z));
    return (worst > 1.0) ? center + v / worst : history;
}

vec3 historyTap(vec2 uv) {
    return textureLod(sampler2D(tex_history, smp), uv, 0.0).rgb;
}

float historyDepthTap(vec2 uv) {
    return textureLod(sampler2D(tex_history, smp_point), uv, 0.0).a;
}

// Five-tap Catmull-Rom. The full kernel is 4x4; collapsing each axis' middle
// pair into one bilinear fetch at a weighted position, and dropping the corner
// groups, brings it to five while keeping the sharpening the centre provides.
vec3 sampleHistoryCatmullRom(vec2 uv, vec2 texel) {
    vec2 size = 1.0 / texel;
    vec2 sample_pos = uv * size;
    vec2 tex1 = floor(sample_pos - 0.5) + 0.5;
    vec2 f = sample_pos - tex1;

    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);

    vec2 w12 = w1 + w2;
    vec2 off12 = w2 / w12;

    vec2 pos0 = (tex1 - 1.0) * texel;
    vec2 pos3 = (tex1 + 2.0) * texel;
    vec2 pos12 = (tex1 + off12) * texel;

    vec3 result = vec3(0.0);
    float total = 0.0;

    // Track the range of the taps as well as their weighted sum. Catmull-Rom
    // sharpens by giving the outer taps NEGATIVE weight, which means the result
    // can land outside the values it was built from -- overshooting above the
    // brightest tap and undershooting below the darkest.
    //
    // Beside a very bright surface that undershoot is what draws a dark rim.
    // Pre-tonemap the contrast is extreme (an emissive can be 40.0 against a
    // background near 0.1), so the ring is severe rather than subtle, and
    // clamping to zero does not help: the rim is not negative, just far below
    // its neighbours.
    //
    // Bounding the result by its own taps removes the ringing completely while
    // keeping the sharpening everywhere the kernel stays in range -- which is
    // everywhere except these high-contrast edges.
    vec3 tap_min = vec3(1e30);
    vec3 tap_max = vec3(-1e30);
    vec3 t;
    float k;

    k = w12.x * w0.y;
    t = historyTap(vec2(pos12.x, pos0.y));
    result += t * k; total += k; tap_min = min(tap_min, t); tap_max = max(tap_max, t);

    k = w0.x * w12.y;
    t = historyTap(vec2(pos0.x, pos12.y));
    result += t * k; total += k; tap_min = min(tap_min, t); tap_max = max(tap_max, t);

    k = w12.x * w12.y;
    t = historyTap(vec2(pos12.x, pos12.y));
    result += t * k; total += k; tap_min = min(tap_min, t); tap_max = max(tap_max, t);

    k = w3.x * w12.y;
    t = historyTap(vec2(pos3.x, pos12.y));
    result += t * k; total += k; tap_min = min(tap_min, t); tap_max = max(tap_max, t);

    k = w12.x * w3.y;
    t = historyTap(vec2(pos12.x, pos3.y));
    result += t * k; total += k; tap_min = min(tap_min, t); tap_max = max(tap_max, t);

    // Renormalise: dropping the corner groups leaves the weights short of 1.
    return clamp(result / total, tap_min, tap_max);
}

void main() {
    vec3 current = textureLod(sampler2D(tex_color, smp), v_uv, 0.0).rgb;
    float raw_depth = textureLod(sampler2D(tex_depth, smp_point), v_uv, 0.0).r;
    float view_depth = depth_params.y / (raw_depth - depth_params.x);

    if (flags.x > 0.5) {
        frag_color = vec4(current, view_depth);
        return;
    }

    vec2 texel = params.xy;
    bool flip = params.w > 0.5;
    float gamma = max(flags.y, 0.1);

    // First and second moments of the 3x3 neighbourhood, in YCoCg. Mean and
    // standard deviation from these describe the neighbourhood far better than
    // its extremes do.
    vec3 m1 = vec3(0.0);
    vec3 m2 = vec3(0.0);
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec3 c = rgbToYcocg(textureLod(sampler2D(tex_color, smp), v_uv + vec2(x, y) * texel, 0.0).rgb);
            m1 += c;
            m2 += c * c;
        }
    }
    const float inv_n = 1.0 / 9.0;
    vec3 mu = m1 * inv_n;
    vec3 sigma = sqrt(max(m2 * inv_n - mu * mu, vec3(0.0)));
    vec3 box_min = mu - gamma * sigma;
    vec3 box_max = mu + gamma * sigma;

    // Camera reprojection, computed unconditionally. It is a handful of
    // arithmetic operations with no texture fetch, so computing it and then
    // discarding it where a recorded velocity exists is cheaper than the branch
    // would be, and it keeps the control flow flat. Early returns nested inside
    // conditionals are exactly what SPIRV-Cross restructures badly on the HLSL
    // path.
    vec4 world_h = inv_view_proj * vec4(v_ndc, raw_depth, 1.0);
    vec3 world = world_h.xyz / world_h.w;
    vec4 prev_clip = prev_view_proj * vec4(world, 1.0);

    bool camera_valid = prev_clip.w > 0.0;
    vec2 camera_uv = camera_valid
        ? (prev_clip.xy / max(prev_clip.w, 1e-6)) * 0.5 + 0.5
        : v_uv;
    if (flip) camera_uv.y = 1.0 - camera_uv.y;

    vec2 velocity = vec2(0.0);
    if (flags.w > 0.5) {
        velocity = textureLod(sampler2D(tex_velocity, smp_point), v_uv, 0.0).rg;
    }
    bool has_velocity = dot(velocity, velocity) > 1e-12;

    vec2 prev_uv = has_velocity ? (v_uv + velocity) : camera_uv;

    // RGB similarity alone cannot tell whether a history sample belongs to the
    // same surface. This is especially harmful for dark AO and shadowed areas:
    // a stale dark sample can easily fit the current colour neighbourhood and
    // then blur/flood across an edge. History alpha carries the prior frame's
    // view-space depth, while `prev_clip.w` is this static surface's expected
    // view-space depth in that frame (perspectiveFovLh writes view z to w).
    // Moving objects use their velocity vector instead; their expected previous
    // depth cannot be recovered from camera reprojection alone.
    float expected_prev_depth = prev_clip.w;
    float recorded_prev_depth = historyDepthTap(prev_uv);
    float depth_tolerance = max(0.01, abs(expected_prev_depth) * depth_params.z);
    bool depth_valid = abs(recorded_prev_depth - expected_prev_depth) <= depth_tolerance;

    // No history where the camera reprojection was invalid and nothing recorded
    // a velocity, or where the surface was off screen last frame.
    bool usable = (has_velocity || camera_valid)
        && (has_velocity || depth_valid)
        && prev_uv.x >= 0.0 && prev_uv.x <= 1.0
        && prev_uv.y >= 0.0 && prev_uv.y <= 1.0;
    if (!usable) {
        frag_color = vec4(current, view_depth);
        return;
    }

    // Blend between a plain bilinear fetch and the sharpening bicubic.
    //
    // SHARPENING HERE IS A FEEDBACK LOOP: the sharpened result becomes next
    // frame's history, is sharpened again, and so on. Each pass is legal within
    // its own taps, but the overshoot compounds, and after enough frames it
    // shows as a bright rim hugging every silhouette that sits against a darker
    // background -- the halo around a sphere or a post, and the lighter edge
    // along a wall.
    //
    // At full strength the loop wins. At zero, the repeated bilinear resampling
    // softens the image, which is the problem the bicubic was added to solve.
    // Partial strength recovers most of the sharpness while keeping the
    // accumulated overshoot below where it becomes visible.
    float sharpen = clamp(flags.z, 0.0, 1.0);
    vec3 history_bilinear = historyTap(prev_uv);
    vec3 history = (sharpen > 0.001)
        ? mix(history_bilinear, sampleHistoryCatmullRom(prev_uv, texel), sharpen)
        : history_bilinear;
    history = ycocgToRgb(clipToBox(box_min, box_max, rgbToYcocg(history)));

    vec3 a = tonemapWeight(current);
    vec3 b = tonemapWeight(history);
    frag_color = vec4(tonemapUnweight(mix(b, a, params.z)), view_depth);
}
@end

@program taa_resolve vs fs
