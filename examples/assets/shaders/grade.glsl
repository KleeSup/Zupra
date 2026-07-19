//------------------------------------------------------------------------------
//  shaders/grade.glsl
//
//  Colour grading — the workhorse LDR post effect. Every engine ships one
//  (Unreal's Color Grading, Unity's Color Adjustments, Godot's Environment
//  adjustments) because it's how a scene gets its final look without touching
//  materials or lights.
//
//  Unlike chromatic aberration, this affects EVERY pixel including flat areas,
//  so it's also the effect to reach for when you want to verify a post chain is
//  actually running.
//
//  Order matters and follows standard grading practice: exposure (scene-linear
//  scale) -> contrast (pivot around mid-grey) -> saturation (toward luma) ->
//  tint (channel multiply) -> vignette (spatial falloff last, so it darkens the
//  graded result rather than being graded itself).
//
//    fs_params:
//      tone.x = exposure    1.0 neutral, 2.0 twice as bright
//      tone.y = contrast    1.0 neutral, 0 flat grey, 2.0 crushed
//      tone.z = saturation  1.0 neutral, 0 greyscale, 2.0 lurid
//      tone.w = vignette    0.0 off, 1.0 strong corner darkening
//      tint.rgb             per-channel multiply, (1,1,1) neutral
//      tint.w = vignette softness, 0.2 hard ring .. 1.5 gentle
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
layout(binding=0) uniform texture2D u_input;
layout(binding=0) uniform sampler u_smp;

layout(binding=0) uniform fs_params {
    vec4 tone; // exposure, contrast, saturation, vignette strength
    vec4 tint; // rgb multiply, w = vignette softness
};

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec3 c = textureLod(sampler2D(u_input, u_smp), v_uv, 0.0).rgb;

    // Exposure — a straight scale. Applied first so everything downstream sees
    // the intended brightness.
    c *= tone.x;

    // Contrast — pivot around mid-grey so the image gets darker AND brighter
    // rather than just brighter. 0.5 is the right pivot post-tonemap.
    c = (c - 0.5) * tone.y + 0.5;

    // Saturation — interpolate between the luma-only version and full colour.
    // Values above 1 extrapolate past the original, which is what makes a
    // stylised look possible.
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722)); // Rec.709 luma
    c = mix(vec3(l), c, tone.z);

    // Tint — per-channel multiply. Warm with (1.05, 1.0, 0.95), cool by
    // inverting that. Also how you'd fake a colour cast for a mood.
    c *= tint.rgb;

    // Vignette — darken by distance from centre. Aspect-correct so it stays
    // circular on a wide window instead of stretching into an ellipse.
    if (tone.w > 0.0) {
        vec2 d = v_uv - vec2(0.5);
        float r = length(d) * 2.0;
        float v = 1.0 - tone.w * smoothstep(1.0 - max(tint.w, 0.01), 1.0 + max(tint.w, 0.01), r);
        c *= clamp(v, 0.0, 1.0);
    }

    frag_color = vec4(max(c, vec3(0.0)), 1.0);
}
@end

@program grade vs fs
