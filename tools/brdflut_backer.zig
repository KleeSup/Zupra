// tools/gen_brdf.zig
// Runs at build-time to produce resources/brdf.lut
const std = @import("std");

const LUT_SIZE = 512;
const SAMPLE_COUNT = 4096;
const PI = std.math.pi;

pub fn main() !void {
    var pixels: [LUT_SIZE * LUT_SIZE * 2]f16 = undefined;

    for (0..LUT_SIZE) |y| {
        for (0..LUT_SIZE) |x| {
            const NdotV = (@as(f32, @floatFromInt(x)) + 0.5) / @as(f32, LUT_SIZE);
            const roughness = (@as(f32, @floatFromInt(y)) + 0.5) / @as(f32, LUT_SIZE);
            const result = integrateBRDF(NdotV, roughness);
            const idx = (y * LUT_SIZE + x) * 2;
            pixels[idx + 0] = @floatCast(result[0]);
            pixels[idx + 1] = @floatCast(result[1]);
        }
    }

    // Path is relative to where the tool is run from (the project root)
    const file = try std.fs.cwd().createFile("resources/brdf.lut", .{});
    defer file.close();
    try file.writeAll(std.mem.sliceAsBytes(&pixels));

    std.debug.print("BRDF LUT baked: resources/brdf.lut ({d}x{d}, {d} samples)\n", .{ LUT_SIZE, LUT_SIZE, SAMPLE_COUNT });
}

fn integrateBRDF(NdotV: f32, roughness: f32) [2]f32 {
    const V = [3]f32{ @sqrt(1.0 - NdotV * NdotV), 0.0, NdotV };
    var A: f32 = 0.0;
    var B: f32 = 0.0;

    for (0..SAMPLE_COUNT) |i| {
        const Xi = hammersley(i, SAMPLE_COUNT);
        const H = importanceSampleGGX(Xi, roughness);
        const VdotH = V[0] * H[0] + V[1] * H[1] + V[2] * H[2];
        const L = [3]f32{
            2.0 * VdotH * H[0] - V[0],
            2.0 * VdotH * H[1] - V[1],
            2.0 * VdotH * H[2] - V[2],
        };
        const NdotL = @max(L[2], 0.0);
        const NdotH = @max(H[2], 0.0);
        const VdotH_c = @max(VdotH, 0.0);

        if (NdotL > 0.0) {
            const G = geometrySmith(NdotV, NdotL, roughness);
            const G_Vis = (G * VdotH_c) / (NdotH * NdotV + 0.0001);
            const Fc = std.math.pow(f32, 1.0 - VdotH_c, 5.0);
            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    return .{ A / @as(f32, SAMPLE_COUNT), B / @as(f32, SAMPLE_COUNT) };
}

fn hammersley(i: usize, N: u32) [2]f32 {
    return .{ @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(N)), radicalInverse(@intCast(i)) };
}

fn radicalInverse(bits_in: u32) f32 {
    var bits = bits_in;
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    return @as(f32, @floatFromInt(bits)) * 2.3283064365386963e-10;
}

fn geometrySmith(NdotV: f32, NdotL: f32, roughness: f32) f32 {
    const k = (roughness * roughness) / 2.0;
    const ggx1 = NdotV / (NdotV * (1.0 - k) + k);
    const ggx2 = NdotL / (NdotL * (1.0 - k) + k);
    return ggx1 * ggx2;
}

fn importanceSampleGGX(Xi: [2]f32, roughness: f32) [3]f32 {
    const a = roughness * roughness;
    const phi = 2.0 * PI * Xi[0];
    const cosTheta = @sqrt((1.0 - Xi[1]) / (1.0 + (a * a - 1.0) * Xi[1]));
    const sinTheta = @sqrt(1.0 - cosTheta * cosTheta);
    return .{
        sinTheta * @cos(phi),
        sinTheta * @sin(phi),
        cosTheta,
    };
}
