//------------------------------------------------------------------------------
//  shaders/taa_resolve.glsl
//
//  Temporal anti-aliasing resolve.
//
//  THE IDEA. Each frame is rendered with the projection nudged by a fraction of
//  a pixel, so consecutive frames sample the scene at different points inside
//  each pixel. Accumulating them recovers detail no single frame contains --
//  which is why TAA both anti-aliases geometry AND cleans up stochastic effects
//  like GTAO, whose per-pixel noise averages out across the sequence. That
//  second property is the reason it comes before GTAO rather than after.
//
//  REPROJECTION. Last frame's pixel is somewhere else this frame, so the history
//  is looked up at where this surface WAS. Position is reconstructed from depth
//  and pushed through the previous view-projection. This is exact for static
//  geometry and for any camera motion; an object that moved on its own is not
//  accounted for, and relies on the clamp below.
//
//  NEIGHBOURHOOD CLAMPING is what makes it usable rather than a smear. History
//  is only trusted where it agrees with what is actually on screen now: it is
//  clamped into the colour range of the current pixel's 3x3 neighbourhood, so
//  when a surface is newly revealed, or an object moved without a velocity to
//  follow, the stale colour is pulled to something plausible instead of ghosting.
//
//  TONEMAP-WEIGHTED BLENDING. Averaging HDR values directly lets one very bright
//  sample dominate the mean and flicker for several frames afterwards. Weighting
//  each contribution by 1/(1+luma) during the blend and undoing it afterwards
//  averages in a perceptual space instead, which is the standard fix.
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
    // Current inverse view-projection, UNJITTERED.
    mat4 inv_view_proj;
    // Previous frame's view-projection, also unjittered.
    mat4 prev_view_proj;
    // xy = 1 / target size, z = blend weight for the current frame,
    // w = 1 when the render target reads top-left first.
    vec4 params;
    // x = 1 to reset history (first frame, or after a resize/teleport).
    vec4 flags;
};

layout(binding=0) uniform texture2D tex_color;
layout(binding=1) uniform texture2D tex_history;
layout(binding=2) uniform texture2D tex_depth;
layout(binding=0) uniform sampler smp;

in vec2 v_uv;
in vec2 v_ndc;
out vec4 frag_color;

float luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Weight bright samples down before averaging, and undo it after.
vec3 tonemapWeight(vec3 c) {
    return c / (1.0 + luma(c));
}
vec3 tonemapUnweight(vec3 c) {
    return c / max(1.0 - luma(c), 1e-4);
}

void main() {
    vec3 current = textureLod(sampler2D(tex_color, smp), v_uv, 0.0).rgb;

    if (flags.x > 0.5) {
        frag_color = vec4(current, 1.0);
        return;
    }

    vec2 texel = params.xy;
    bool flip = params.w > 0.5;

    // Colour extent of the 3x3 neighbourhood, used to bound the history below.
    vec3 nmin = current;
    vec3 nmax = current;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) continue;
            vec3 c = textureLod(sampler2D(tex_color, smp), v_uv + vec2(x, y) * texel, 0.0).rgb;
            nmin = min(nmin, c);
            nmax = max(nmax, c);
        }
    }

    // Where was this surface last frame?
    float raw_depth = textureLod(sampler2D(tex_depth, smp), v_uv, 0.0).r;
    vec4 world_h = inv_view_proj * vec4(v_ndc, raw_depth, 1.0);
    vec3 world = world_h.xyz / world_h.w;

    vec4 prev_clip = prev_view_proj * vec4(world, 1.0);
    if (prev_clip.w <= 0.0) {
        // Behind last frame's eye: no history to reproject from.
        frag_color = vec4(current, 1.0);
        return;
    }
    vec2 prev_ndc = prev_clip.xy / prev_clip.w;
    vec2 prev_uv = prev_ndc * 0.5 + 0.5;
    if (flip) prev_uv.y = 1.0 - prev_uv.y;

    // Off-screen last frame: this surface is newly visible and has no history.
    if (prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0) {
        frag_color = vec4(current, 1.0);
        return;
    }

    vec3 history = textureLod(sampler2D(tex_history, smp), prev_uv, 0.0).rgb;

    // Trust history only as far as it agrees with what is on screen now.
    // Disocclusions and unaccounted object motion both land outside this box,
    // and clamping converts what would be a lingering ghost into at most a few
    // frames of slightly stale colour.
    history = clamp(history, nmin, nmax);

    // Blend in tonemapped space so one bright sample cannot dominate the mean.
    vec3 a = tonemapWeight(current);
    vec3 b = tonemapWeight(history);
    vec3 result = tonemapUnweight(mix(b, a, params.z));

    frag_color = vec4(result, 1.0);
}
@end

@program taa_resolve vs fs
