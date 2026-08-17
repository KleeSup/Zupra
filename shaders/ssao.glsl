//------------------------------------------------------------------------------
//  shaders/ssao.glsl
//
//  GTAO -- Ground Truth Ambient Occlusion (Jimenez et al., 2016).
//
//  WHY THIS RATHER THAN HEMISPHERE SAMPLING. The older method scattered points
//  and counted how many landed behind geometry: a Monte Carlo estimate of a
//  quantity that is not quite the cosine-weighted visibility integral the
//  rendering equation asks for. GTAO finds the exact horizon angles along a few
//  slices through the hemisphere and solves that integral across them in closed
//  form. It converges to a ray-traced reference instead of plateauing short of
//  it, needs far fewer samples for the same cleanliness, and -- the property
//  that matters most in motion -- varies SMOOTHLY as the camera moves, because
//  horizon angles are continuous where hit-counting is discrete and pops.
//
//  THE MARCH IS IN SCREEN PIXELS, NOT WORLD UNITS. This is the detail that
//  decides whether the effect works at all. A world-space radius projects to
//  fewer and fewer pixels with distance, so a fixed world radius eventually
//  spans less than a texel: every step lands on the same pixel, no horizon is
//  ever found, and the buffer comes back uniformly unoccluded. Converting the
//  radius to pixels and CLAMPING that to a sane range fixes both ends -- distant
//  geometry keeps a usable search width, and geometry right against the camera
//  cannot explode into a full-screen march.
//
//  VIEW SPACE for the geometry. For a perspective projection, moving between a
//  view-space point and its screen position is two multiplies, and view depth
//  comes straight from the depth buffer as z = B/(depth - A). No matrices in any
//  inner loop.
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
layout(binding=0) uniform ssao_params {
    // World -> view, for rotating the G-buffer's world-space normal.
    mat4 view;
    // x = proj[0][0], y = proj[1][1], z = 1/x, w = 1/y.
    vec4 proj_xy;
    // x, y = the A and B of z_view = B / (depth - A).
    // z = 1 when the render target reads top-left first. w unused.
    vec4 depth_lin;
    // x = radius (world units), y = intensity, z = thickness, w = falloff power.
    vec4 params;
    // x = angle bias in radians, yzw unused.
    vec4 bias;
    // x = slices, y = steps per direction, zw = render target size in pixels.
    vec4 counts;
    // x = frame index for temporal jitter, y = 1 when temporal jitter is on.
    vec4 temporal;
};

layout(binding=0) uniform texture2D tex_depth;
layout(binding=1) uniform texture2D tex_normal;
layout(binding=0) uniform sampler smp;

in vec2 v_uv;
in vec2 v_ndc;
out vec4 frag_color;

const float PI = 3.14159265359;
const float HALF_PI = 1.57079632679;

// Search width limits, in pixels. The lower bound keeps distant geometry from
// degenerating into a sub-pixel march that can never find a horizon; the upper
// bound caps cost and cache misses when a surface is right against the lens.
const float MIN_RADIUS_PX = 4.0;
const float MAX_RADIUS_PX = 96.0;

float viewZ(float raw) {
    return depth_lin.y / (raw - depth_lin.x);
}

float sampleViewZ(vec2 uv) {
    return viewZ(textureLod(sampler2D(tex_depth, smp), uv, 0.0).r);
}

vec3 viewPos(vec2 ndc, float z) {
    return vec3(ndc.x * z * proj_xy.z, ndc.y * z * proj_xy.w, z);
}

// Interleaved gradient noise. Its error is high-frequency, which is exactly what
// the blur removes; a low-frequency pattern would survive it.
float igNoise(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

// TEMPORAL JITTER. Spatial noise alone decorrelates neighbouring PIXELS but
// gives each pixel the same probe directions every frame -- so a temporal filter
// averaging across frames would average the same value repeatedly and gain
// nothing. Cycling the slice rotation and the march offset makes successive
// frames sample genuinely different directions, and that is what lets TAA
// resolve GTAO's error rather than merely smearing it.
//
// Two sequences of coprime length (6 and 4) so the pair does not repeat for 12
// frames. These are the offsets from the original GTAO presentation.
float temporalRotation(int frame) {
    // 60, 300, 180, 240, 120, 0 degrees, as turns.
    int i = frame % 6;
    if (i == 0) return 1.0 / 6.0;
    if (i == 1) return 5.0 / 6.0;
    if (i == 2) return 3.0 / 6.0;
    if (i == 3) return 4.0 / 6.0;
    if (i == 4) return 2.0 / 6.0;
    return 0.0;
}

float temporalOffset(int frame) {
    int i = frame % 4;
    if (i == 0) return 0.0;
    if (i == 1) return 0.5;
    if (i == 2) return 0.25;
    return 0.75;
}

void main() {
    vec4 nrm = textureLod(sampler2D(tex_normal, smp), v_uv, 0.0);
    // Background is fully open. Writing 1.0 rather than discarding keeps the
    // target defined everywhere so the blur never reads undefined texels.
    if (nrm.a < 0.5) {
        frag_color = vec4(1.0);
        return;
    }

    vec2 screen = max(counts.zw, vec2(1.0));
    float z = sampleViewZ(v_uv);
    vec3 P = viewPos(v_ndc, z);
    // Rows arrive as columns, so view * v is the row-vector transform. w = 0
    // because a normal is a direction and translation must not apply.
    vec3 N = normalize((view * vec4(normalize(nrm.xyz), 0.0)).xyz);
    // Toward the eye. Horizons are measured against this.
    vec3 V = normalize(-P);

    float radius = params.x;
    float thickness = max(params.z, 1e-3);
    int slices = int(counts.x);
    int steps = int(counts.y);
    bool flip = depth_lin.z > 0.5;

    // World radius -> pixels. A world length r at depth z spans r*m00/z in NDC,
    // and NDC spans 2 across the screen, so r*m00/z * 0.5 * width in pixels.
    // The pixel budget bounds the number of taps, NOT the world extent searched.
    //
    // Clamping the extent, as this did before, silently changed what the effect
    // measures depending on where the camera stood: at one unit away a requested
    // radius of 1.0 wanted 624 pixels, clamped to 96, and covered only 0.15
    // world units, while at ten units away it covered the full metre. Same
    // geometry, different answer, which is exactly the view dependence that made
    // this so hard to pin down.
    float wanted_px = radius * proj_xy.x * 0.5 * screen.x / max(z, 1e-4);
    float radius_px = clamp(wanted_px, MIN_RADIUS_PX, MAX_RADIUS_PX);
    // How much of the requested radius the pixel budget can actually reach. Below
    // 1 when close to a surface: fewer taps over the same world distance, which
    // costs precision rather than changing the measurement.
    float reach = clamp(radius_px / max(wanted_px, 1e-4), 0.0, 1.0);
    // Falloff is anchored to the world radius the caller asked for, never to the
    // clamped pixel radius, so a sample at a given world distance is discounted
    // by the same amount from every viewpoint.
    float inv_r = 1.0 / max(radius, 1e-4);

    // Two independent jitters, each combining a spatial and a temporal term.
    // Spatial decorrelates neighbouring pixels; temporal decorrelates successive
    // frames so the accumulation actually converges.
    int frame = int(temporal.x);
    bool use_temporal = temporal.y > 0.5;
    float noise = igNoise(gl_FragCoord.xy);
    float slice_jitter = noise;
    float step_jitter = fract(noise * 7.0);
    if (use_temporal) {
        slice_jitter = fract(slice_jitter + temporalRotation(frame));
        step_jitter = fract(step_jitter + temporalOffset(frame));
    }

    // Step length in WORLD units. `reach` scales it down when the pixel budget
    // cannot cover the full radius, so the taps stay usefully spaced rather than
    // landing on the same texel.
    float step_ws = (radius * reach) / float(steps);
    float visibility = 0.0;
    // Running total of what each slice would contribute with NOTHING occluding
    // it. Dividing by this rather than by the slice count is what makes an open
    // surface come out at exactly 1 regardless of how it is tilted -- see the
    // note where it is accumulated below.
    float open_total = 0.0;

    for (int s = 0; s < slices; s++) {
        // Slices span half a turn; marching both ways covers the rest.
        float phi = PI * (float(s) + slice_jitter) / float(slices);
        vec2 dir_px = vec2(cos(phi), sin(phi));
        // Unit direction in NDC, aspect-corrected so a slice at 45 degrees
        // covers the same world distance as one along an axis. Scaled per sample
        // by the world offset below.
        vec2 dir_ndc_unit = vec2(dir_px.x, dir_px.y * proj_xy.y / proj_xy.x);

        // The slice's plane contains V and the screen direction. Projecting N
        // into it is what ties the 2D horizon search back to the 3D surface.
        vec3 dir3 = vec3(dir_px, 0.0);
        vec3 axis = normalize(cross(dir3, V));
        vec3 proj_n = N - axis * dot(N, axis);
        float proj_len = length(proj_n);
        if (proj_len < 1e-4) continue; // slice edge-on to the normal: no arc

        float cos_n = clamp(dot(proj_n, V) / proj_len, -1.0, 1.0);
        vec3 ortho = dir3 - V * dot(dir3, V);
        float n_angle = sign(dot(ortho, proj_n)) * acos(cos_n);

        // cos of the highest angle found each way. -1 means nothing above the
        // horizon, i.e. fully open.
        float cos_h1 = -1.0;
        float cos_h2 = -1.0;

        for (int j = 0; j < steps; j++) {
            float t_ws = (float(j) + step_jitter + 0.5) * step_ws;
            // World offset -> NDC at this pixel's depth. Doing the conversion
            // here rather than marching in pixels is what keeps the searched
            // volume the same from every distance.
            float t_ndc = t_ws * proj_xy.x / max(z, 1e-4);

            for (int side = 0; side < 2; side++) {
                float sgn = (side == 0) ? 1.0 : -1.0;
                vec2 ndc_s = v_ndc + dir_ndc_unit * (t_ndc * sgn);
                if (abs(ndc_s.x) > 1.0 || abs(ndc_s.y) > 1.0) continue;

                vec2 uv_s = ndc_s * 0.5 + 0.5;
                if (flip) uv_s.y = 1.0 - uv_s.y;

                vec3 S = viewPos(ndc_s, sampleViewZ(uv_s));
                vec3 D = S - P;
                float len = length(D);
                if (len < 1e-5) continue;

                float cos_h = dot(D, V) / len;

                // THE HALO FIX. A sample far from the surface is usually a
                // different object seen past an edge, not a wall beside us --
                // and counting it as an occluder is exactly what makes horizon
                // methods draw dark rings around silhouettes. Fading its
                // influence back toward "open" with distance removes the ring
                // without giving up nearby occlusion.
                float falloff = clamp(1.0 - (len * inv_r) * thickness, 0.0, 1.0);
                cos_h = mix(-1.0, cos_h, falloff);

                if (side == 0) {
                    cos_h1 = max(cos_h1, cos_h);
                } else {
                    cos_h2 = max(cos_h2, cos_h);
                }
            }
        }

        // Horizons as signed angles either side of V, clamped into the
        // hemisphere about the projected normal: a surface cannot occlude itself
        // from behind.
        float h1 = -acos(clamp(cos_h1, -1.0, 1.0));
        float h2 = acos(clamp(cos_h2, -1.0, 1.0));

        // ANGLE BIAS -- push each horizon a few degrees back toward the tangent
        // plane before using it. h1 is the negative side and h2 the positive
        // one, so "back" means more negative and more positive respectively.
        //
        // On a genuinely flat surface every sample should report the horizon
        // exactly at the tangent plane. Depth reconstruction is not exact:
        // precision and the point-sampled depth buffer scatter samples a hair
        // either side of the true plane, and the search takes max() over them.
        // The maximum of N noisy values drifts upward with N -- so flat surfaces
        // darken in proportion to STEP COUNT, which is wrong and is the clearest
        // symptom of the problem.
        //
        // A real occluder rises well clear of the tangent, so discarding the
        // last few degrees costs almost nothing in a crease while removing the
        // noise floor completely. The hemisphere clamp is applied AFTER, so an
        // unoccluded surface still lands exactly on the tangent and the
        // normalisation below still yields 1.
        h1 -= bias.x;
        h2 += bias.x;
        h1 = n_angle + max(h1 - n_angle, -HALF_PI);
        h2 = n_angle + min(h2 - n_angle, HALF_PI);

        // The analytic inner integral -- the step that makes GTAO exact within a
        // slice. No sampling: a closed form for cosine-weighted visibility over
        // the arc between the two horizons. With no occluders this evaluates to
        // exactly 1, so an open surface is untouched.
        float sin_n = sin(n_angle);
        float arc = 0.25 * (-cos(2.0 * h1 - n_angle) + cos_n + 2.0 * h1 * sin_n)
                  + 0.25 * (-cos(2.0 * h2 - n_angle) + cos_n + 2.0 * h2 * sin_n);

        // Weighted by how much of N survived projection into this slice: a slice
        // nearly edge-on to the normal should count for little.
        visibility += proj_len * arc;

        // NORMALISATION. With no occluders at all, h1 and h2 sit at the tangent
        // plane and the arc integral collapses to cos(n) + n*sin(n) -- which is
        // 1 only when the surface faces the eye square-on, and grows as it
        // tilts (1.117 at n = 0.5 rad, more at grazing angles).
        //
        // Dividing by the slice count therefore leaves a tilted surface with
        // SPARE visibility above 1. It absorbs real occlusion and still clamps
        // to 1.0, while the flatter surface beside it starts at exactly 1.0 and
        // darkens immediately. That difference draws a rim that never darkens
        // around every silhouette in the scene.
        //
        // Dividing by the open value instead makes any unoccluded surface come
        // out at exactly 1 whatever its orientation, so occlusion is measured
        // against a consistent baseline.
        float sin_full = sin(n_angle);
        open_total += proj_len * (cos_n + n_angle * sin_full);
    }

    visibility = clamp(visibility / max(open_total, 1e-4), 0.0, 1.0);

    float ao = 1.0 - (1.0 - visibility) * params.y;
    frag_color = vec4(pow(clamp(ao, 0.0, 1.0), params.w));
}
@end

@program ssao vs fs
