//------------------------------------------------------------------------------
// shaders/ssao.glsl
//
// GTAO -- Ground Truth Ambient Occlusion.
//
// This version keeps the original structure of this shader but fixes:
//
//   1. Positive/negative horizon assignment.
//   2. Y-flipped depth reconstruction.
//   3. Visibility normalization.
//   4. Open-surface normalization for tilted surfaces.
//   5. Slice skipping / normalization edge cases.
//   6. Sample falloff handling.
//   7. Numerical safety around zero-length vectors.
//
//------------------------------------------------------------------------------

@vs vs

in vec2 pos;
in vec2 uv;

out vec2 v_uv;
out vec2 v_ndc;

void main()
{
    gl_Position = vec4(pos, 0.0, 1.0);

    v_uv  = uv;
    v_ndc = pos;
}

@end


@fs fs

layout(binding=0) uniform ssao_params
{
    // World -> view.
    mat4 view;

    // x = proj[0][0]
    // y = proj[1][1]
    // z = 1 / proj[0][0]
    // w = 1 / proj[1][1]
    vec4 proj_xy;

    // x = A
    // y = B
    //
    // viewZ = B / (depth - A)
    //
    // z = 1 when depth needs a vertical flip.
    vec4 depth_lin;

    // x = radius in world units
    // y = AO intensity
    // z = thickness/falloff multiplier
    // w = final AO power
    vec4 params;

    // x = angular bias in radians
    vec4 bias;

    // x = number of slices
    // y = steps per direction
    // z = render target width
    // w = render target height
    vec4 counts;

    // x = frame index
    // y = temporal jitter enabled
    vec4 temporal;
};


layout(binding=0) uniform texture2D tex_depth;
layout(binding=1) uniform texture2D tex_normal;
layout(binding=0) uniform sampler smp;

in vec2 v_uv;
in vec2 v_ndc;

out vec4 frag_color;


const float PI      = 3.14159265359;
const float HALF_PI = 1.57079632679;

const float MIN_RADIUS_PX = 4.0;
const float MAX_RADIUS_PX = 96.0;


//------------------------------------------------------------------------------
// Depth reconstruction
//------------------------------------------------------------------------------

float viewZ(float raw)
{
    return depth_lin.y / (raw - depth_lin.x);
}


float sampleViewZ(vec2 uv)
{
    return viewZ(
        textureLod(
            sampler2D(tex_depth, smp),
            uv,
            0.0
        ).r
    );
}


//------------------------------------------------------------------------------
// View-space position.
//
// ndc is expected to use the same vertical convention as the reconstructed
// view-space coordinates.
//
// The depth texture itself may have a different Y orientation, which is why
// the UV used to SAMPLE depth can be flipped independently.
//------------------------------------------------------------------------------

vec3 viewPos(vec2 ndc, float z)
{
    return vec3(
        ndc.x * z * proj_xy.z,
        ndc.y * z * proj_xy.w,
        z
    );
}


//------------------------------------------------------------------------------
// Interleaved gradient noise
//------------------------------------------------------------------------------

float igNoise(vec2 p)
{
    return fract(
        52.9829189 *
        fract(
            dot(
                p,
                vec2(0.06711056, 0.00583715)
            )
        )
    );
}


//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

void main()
{
    //--------------------------------------------------------------------------
    // Read normal
    //--------------------------------------------------------------------------

    vec4 nrm = textureLod(
        sampler2D(tex_normal, smp),
        v_uv,
        0.0
    );

    // Background.
    if (nrm.a < 0.5)
    {
        frag_color = vec4(1.0);
        return;
    }


    //--------------------------------------------------------------------------
    // Basic setup
    //--------------------------------------------------------------------------

    vec2 screen = max(counts.zw, vec2(1.0));

    float z = sampleViewZ(v_uv);

    if (abs(z) < 1e-5)
    {
        frag_color = vec4(1.0);
        return;
    }

    vec3 P = viewPos(v_ndc, z);


    // Transform world-space normal to view-space.
    //
    // NOTE:
    // This assumes the matrix supplied as `view` is the world->view matrix
    // expected by your existing renderer.
    //
    vec3 N = normalize(
        (view * vec4(normalize(nrm.xyz), 0.0)).xyz
    );


    // Vector from pixel toward camera.
    vec3 V = normalize(-P);


    float radius   = max(params.x, 0.0);
    float intensity = max(params.y, 0.0);
    float thickness = max(params.z, 1e-3);
    float ao_power = max(params.w, 1e-3);

    int slices = max(int(counts.x), 1);
    int steps  = max(int(counts.y), 1);

    bool flip = depth_lin.z > 0.5;


    //--------------------------------------------------------------------------
    // Convert world radius -> screen pixels
    //--------------------------------------------------------------------------

    float radius_px =
        radius *
        proj_xy.x *
        0.5 *
        screen.x /
        max(abs(z), 1e-4);

    radius_px = clamp(
        radius_px,
        MIN_RADIUS_PX,
        MAX_RADIUS_PX
    );


    //--------------------------------------------------------------------------
    // Convert the actual pixel radius back to world space.
    //
    // This keeps the search radius and falloff radius consistent when the
    // screen-space radius gets clamped.
    //--------------------------------------------------------------------------

    float radius_ws =
        radius_px *
        2.0 *
        max(abs(z), 1e-4) /
        max(proj_xy.x * screen.x, 1e-4);

    radius_ws = max(radius_ws, 1e-4);

    float inv_radius_ws = 1.0 / radius_ws;


    //--------------------------------------------------------------------------
    // Temporal/spatial jitter
    //--------------------------------------------------------------------------

    bool use_temporal = temporal.z > 0.5;

    float noise = igNoise(gl_FragCoord.xy);

    float slice_jitter = noise;
    float step_jitter = fract(noise * 7.0);

    if (use_temporal) {
        slice_jitter = fract(slice_jitter + temporal.x);
        step_jitter = fract(step_jitter + temporal.y);
    }


    //--------------------------------------------------------------------------
    // March distance
    //--------------------------------------------------------------------------

    float step_px =
        radius_px /
        float(steps);


    //--------------------------------------------------------------------------
    // Visibility accumulation.
    //
    // `visibility` = actual visibility integral.
    //
    // `open_total` = what that same integral would be if there were no
    // occluders at all.
    //
    // Dividing the two gives:
    //
    //     open surface -> 1.0
    //
    // regardless of view angle.
    //--------------------------------------------------------------------------

    float visibility = 0.0;
    float open_total = 0.0;

    int valid_slices = 0;


    //--------------------------------------------------------------------------
    // Slice loop
    //--------------------------------------------------------------------------

    for (int s = 0; s < slices; s++)
    {
        //----------------------------------------------------------------------
        // Slice direction
        //
        // Half-turn is enough because each slice is marched in both directions.
        //----------------------------------------------------------------------

        float phi =
            PI *
            (float(s) + slice_jitter) /
            float(slices);

        vec2 dir_px =
            vec2(
                cos(phi),
                sin(phi)
            );

        vec2 dir_ndc =
            dir_px *
            2.0 /
            screen;


        //----------------------------------------------------------------------
        // Build slice plane.
        //
        // The plane contains:
        //
        //   V
        //
        // and the screen-space march direction.
        //----------------------------------------------------------------------

        vec3 dir3 =
            vec3(
                dir_px,
                0.0
            );

        vec3 axis_raw =
            cross(
                dir3,
                V
            );

        float axis_len =
            length(axis_raw);

        if (axis_len < 1e-5)
            continue;

        vec3 axis =
            axis_raw /
            axis_len;


        //----------------------------------------------------------------------
        // Project the normal into the slice plane.
        //----------------------------------------------------------------------

        vec3 proj_n =
            N -
            axis *
            dot(N, axis);

        float proj_len =
            length(proj_n);

        if (proj_len < 1e-5)
            continue;

        proj_n /= proj_len;


        //----------------------------------------------------------------------
        // Normal angle inside the slice.
        //----------------------------------------------------------------------

        float cos_n =
            clamp(
                dot(proj_n, V),
                -1.0,
                1.0
            );

        vec3 ortho =
            dir3 -
            V *
            dot(dir3, V);

        float ortho_len =
            length(ortho);

        float sign_n = 1.0;

        if (ortho_len > 1e-5)
        {
            sign_n =
                sign(
                    dot(
                        ortho,
                        proj_n
                    )
                );

            if (abs(sign_n) < 0.5)
                sign_n = 1.0;
        }

        float n_angle =
            sign_n *
            acos(cos_n);


        //----------------------------------------------------------------------
        // Horizon cosines.
        //
        // IMPORTANT:
        //
        // side == 0 marches in +dir
        // side == 1 marches in -dir
        //
        // We keep those meanings explicit instead of relying on ambiguous
        // h1/h2 names.
        //----------------------------------------------------------------------

        float horizonCosPositive =
            cos(
                n_angle +
                HALF_PI
            );

        float horizonCosNegative =
            cos(
                n_angle -
                HALF_PI
            );


        //----------------------------------------------------------------------
        // Horizon search
        //----------------------------------------------------------------------

        for (int j = 0; j < steps; j++)
        {
            float t_px =
                (
                    float(j) +
                    step_jitter +
                    0.5
                ) *
                step_px;


            //------------------------------------------------------------------
            // March both sides of the slice.
            //------------------------------------------------------------------

            for (int side = 0; side < 2; side++)
            {
                float sgn =
                    (side == 0)
                    ? 1.0
                    : -1.0;


                //----------------------------------------------------------------
                // IMPORTANT:
                //
                // ndc_s is used to reconstruct the geometric position.
                //
                // uv_s is separately converted to the depth texture's
                // orientation.
                //----------------------------------------------------------------

                vec2 sample_ndc =
                    v_ndc +
                    dir_ndc *
                    (t_px * sgn);


                //----------------------------------------------------------------
                // Outside screen.
                //----------------------------------------------------------------

                if (sample_ndc.x < -1.0 ||
                    sample_ndc.x >  1.0 ||
                    sample_ndc.y < -1.0 ||
                    sample_ndc.y >  1.0)
                {
                    continue;
                }


                //----------------------------------------------------------------
                // Convert NDC -> texture UV.
                //
                // This is the Y-flip fix:
                //
                // The sampled depth location may be flipped, but the
                // reconstructed view-space position must use the corresponding
                // NDC location.
                //----------------------------------------------------------------

                vec2 sample_uv =
                    sample_ndc *
                    0.5 +
                    0.5;

                if (flip)
                {
                    sample_uv.y =
                        1.0 -
                        sample_uv.y;
                }


                //----------------------------------------------------------------
                // Sample depth.
                //----------------------------------------------------------------

                float sample_z =
                    sampleViewZ(sample_uv);


                //----------------------------------------------------------------
                // Reconstruct sample position.
                //
                // Use sample_ndc here, NOT the flipped UV.
                //----------------------------------------------------------------

                vec3 S =
                    viewPos(
                        sample_ndc,
                        sample_z
                    );


                //----------------------------------------------------------------
                // Sample delta.
                //----------------------------------------------------------------

                vec3 D =
                    S -
                    P;

                float len =
                    length(D);

                if (len < 1e-5)
                    continue;


                //----------------------------------------------------------------
                // Horizon direction.
                //----------------------------------------------------------------

                vec3 horizonVec =
                    D /
                    len;


                //----------------------------------------------------------------
                // Horizon cosine.
                //----------------------------------------------------------------

                float cos_h =
                    clamp(
                        dot(
                            horizonVec,
                            V
                        ),
                        -1.0,
                        1.0
                    );


                //----------------------------------------------------------------
                // Distance falloff.
                //
                // Samples near the effect radius fade toward the default
                // horizon rather than continuing to behave as strong
                // occluders.
                //----------------------------------------------------------------

                float distance01 =
                    len *
                    inv_radius_ws;

                float falloff =
                    clamp(
                        1.0 -
                        distance01 *
                        thickness,
                        0.0,
                        1.0
                    );


                //----------------------------------------------------------------
                // Interpolate toward the open horizon.
                //
                // This is equivalent in spirit to the falloff treatment in
                // XeGTAO: distant samples progressively stop affecting the
                // horizon.
                //----------------------------------------------------------------

                if (side == 0)
                {
                    cos_h =
                        mix(
                            horizonCosPositive,
                            cos_h,
                            falloff
                        );
                }
                else
                {
                    cos_h =
                        mix(
                            horizonCosNegative,
                            cos_h,
                            falloff
                        );
                }


                //----------------------------------------------------------------
                // Update the appropriate horizon.
                //
                // POSITIVE screen direction -> positive horizon
                // NEGATIVE screen direction -> negative horizon
                //----------------------------------------------------------------

                if (side == 0)
                {
                    horizonCosPositive =
                        max(
                            horizonCosPositive,
                            cos_h
                        );
                }
                else
                {
                    horizonCosNegative =
                        max(
                            horizonCosNegative,
                            cos_h
                        );
                }
            }
        }


        //----------------------------------------------------------------------
        // Convert horizon cosines to signed horizon angles.
        //
        // Positive side:
        //
        //     +acos(...)
        //
        // Negative side:
        //
        //     -acos(...)
        //
        // This ordering is important.
        //----------------------------------------------------------------------

        float h_negative =
            -acos(
                clamp(
                    horizonCosNegative,
                    -1.0,
                    1.0
                )
            );

        float h_positive =
            acos(
                clamp(
                    horizonCosPositive,
                    -1.0,
                    1.0
                )
            );


        //----------------------------------------------------------------------
        // Angular bias.
        //
        // Push the two horizons outward away from the normal:
        //
        // negative -> more negative
        // positive -> more positive
        //----------------------------------------------------------------------

        float angle_bias =
            max(
                bias.x,
                0.0
            );

        h_negative -= angle_bias;
        h_positive += angle_bias;


        //----------------------------------------------------------------------
        // Clamp horizons to the hemisphere around the projected normal.
        //----------------------------------------------------------------------

        h_negative =
            n_angle +
            clamp(
                h_negative - n_angle,
                -HALF_PI,
                HALF_PI
            );

        h_positive =
            n_angle +
            clamp(
                h_positive - n_angle,
                -HALF_PI,
                HALF_PI
            );


        //----------------------------------------------------------------------
        // Analytic GTAO arc integral.
        //
        // This is the closed-form cosine-weighted visibility integral.
        //----------------------------------------------------------------------

        float sin_n =
            sin(n_angle);


        float arc_negative =
            0.25 *
            (
                -cos(
                    2.0 *
                    h_negative -
                    n_angle
                )
                +
                cos_n
                +
                2.0 *
                h_negative *
                sin_n
            );


        float arc_positive =
            0.25 *
            (
                -cos(
                    2.0 *
                    h_positive -
                    n_angle
                )
                +
                cos_n
                +
                2.0 *
                h_positive *
                sin_n
            );


        float arc =
            arc_negative +
            arc_positive;


        //----------------------------------------------------------------------
        // Actual visibility contribution.
        //----------------------------------------------------------------------

        float slice_visibility =
            proj_len *
            arc;

        visibility +=
            slice_visibility;


        //----------------------------------------------------------------------
        // Calculate the EXACT same integral for a completely open slice.
        //
        // With:
        //
        //   h0 = n - PI/2
        //   h1 = n + PI/2
        //
        // the full open arc simplifies to:
        //
        //   cos(n) + n*sin(n)
        //
        // This gives us an orientation-independent baseline.
        //----------------------------------------------------------------------

        float open_arc =
            cos_n +
            n_angle *
            sin_n;


        open_total +=
            proj_len *
            open_arc;


        valid_slices++;
    }


    //--------------------------------------------------------------------------
    // Visibility normalization
    //
    // THIS is the important part you asked to keep.
    //
    // Instead of:
    //
    //     visibility /= slices;
    //
    // we divide by the amount of visibility that the exact same set of slices
    // would have if nothing occluded them.
    //
    // Therefore:
    //
    //     no occluder -> 1.0
    //
    // independent of the surface's orientation.
    //----------------------------------------------------------------------------

    if (valid_slices > 0 &&
        open_total > 1e-5)
    {
        visibility =
            visibility /
            open_total;
    }
    else
    {
        visibility = 1.0;
    }


    //---------------------------------------------------------------------------
    // Numerical clamp.
    //---------------------------------------------------------------------------

    visibility =
        clamp(
            visibility,
            0.0,
            1.0
        );


    //--------------------------------------------------------------------------
    // Convert visibility -> AO.
    //
    // intensity = 1:
    //
    //     AO = visibility
    //
    // intensity = 0:
    //
    //     AO = 1
    //
    //--------------------------------------------------------------------------

    float ao =
        1.0 -
        (
            1.0 -
            visibility
        ) *
        intensity;


    ao =
        clamp(
            ao,
            0.0,
            1.0
        );


    //--------------------------------------------------------------------------
    // Final response curve.
    //----------------------------------------------------------------------

    ao =
        pow(
            ao,
            ao_power
        );


    //---------------------------------------------------------------------------
    // Output.
    //---------------------------------------------------------------------------

    frag_color =
        vec4(
            ao,
            ao,
            ao,
            1.0
        );
}

@end


@program ssao vs fs