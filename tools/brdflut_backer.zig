//! Host-side generator for a raw RG16F split-sum BRDF LUT.
//!
//! Build.zig invokes this only when a project asks for non-default bake
//! settings. Output is supplied by the build graph and lives in Zig's cache;
//! this tool must never mutate the package source tree.
const std = @import("std");

const PI = std.math.pi;

const Options = struct {
    resolution: u32,
    sample_count: u32,
    output_path: []const u8,
};

pub fn main(init: std.process.Init) !void {
    // This short-lived host tool does one allocation for its pixel buffer.
    // Use Zig's process allocator rather than relying on a debug-only allocator
    // API that has changed across Zig development versions.
    const allocator = init.gpa;

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    const program_name = args.next() orelse "zupra_brdf_lut_baker";
    const options = parseArgs(&args) catch |err| {
        std.debug.print(
            "usage: {s} --resolution <pixels> --samples <count> --output <path>\n",
            .{program_name},
        );
        return err;
    };

    try validateOptions(options);

    const resolution: usize = options.resolution;
    const texel_count = try std.math.mul(usize, resolution, resolution);
    const component_count = try std.math.mul(usize, texel_count, 2);
    const pixels = try allocator.alloc(f16, component_count);
    defer allocator.free(pixels);

    for (0..resolution) |y| {
        for (0..resolution) |x| {
            const n_dot_v = (@as(f32, @floatFromInt(x)) + 0.5) / @as(f32, @floatFromInt(options.resolution));
            const roughness = (@as(f32, @floatFromInt(y)) + 0.5) / @as(f32, @floatFromInt(options.resolution));
            const result = integrateBRDF(n_dot_v, roughness, options.sample_count);
            const idx = (y * resolution + x) * 2;
            pixels[idx + 0] = @floatCast(result[0]);
            pixels[idx + 1] = @floatCast(result[1]);
        }
    }

    const file = try std.Io.Dir.cwd().createFile(init.io, options.output_path, .{});
    defer file.close(init.io);
    try file.writeStreamingAll(init.io, std.mem.sliceAsBytes(pixels));

    // Do not print a success message here. `std.debug.print` writes to stderr,
    // which Zig's build runner correctly preserves as a diagnostic for a
    // cache-producing command even when its exit code is zero.
}

fn parseArgs(args: *std.process.Args.Iterator) !Options {
    var resolution: ?u32 = null;
    var sample_count: ?u32 = null;
    var output_path: ?[]const u8 = null;

    while (args.next()) |flag_z| {
        const flag: []const u8 = flag_z;
        const value_z = args.next() orelse return error.MissingOptionValue;
        const value: []const u8 = value_z;

        if (std.mem.eql(u8, flag, "--resolution")) {
            resolution = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--samples")) {
            sample_count = try std.fmt.parseInt(u32, value, 10);
        } else if (std.mem.eql(u8, flag, "--output")) {
            output_path = value;
        } else {
            return error.UnknownOption;
        }
    }

    return .{
        .resolution = resolution orelse return error.MissingResolution,
        .sample_count = sample_count orelse return error.MissingSampleCount,
        .output_path = output_path orelse return error.MissingOutputPath,
    };
}

fn validateOptions(options: Options) !void {
    if (options.resolution == 0 or options.resolution > 4096) return error.InvalidResolution;
    if (options.sample_count == 0 or options.sample_count > 65_536) return error.InvalidSampleCount;
    if (options.output_path.len == 0) return error.InvalidOutputPath;
}

fn integrateBRDF(n_dot_v: f32, roughness: f32, sample_count: u32) [2]f32 {
    const V = [3]f32{ @sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v };
    var A: f32 = 0.0;
    var B: f32 = 0.0;

    var i: u32 = 0;
    while (i < sample_count) : (i += 1) {
        const Xi = hammersley(i, sample_count);
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
            const G = geometrySmith(n_dot_v, NdotL, roughness);
            const G_Vis = (G * VdotH_c) / (NdotH * n_dot_v + 0.0001);
            const Fc = std.math.pow(f32, 1.0 - VdotH_c, 5.0);
            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    const samples_f: f32 = @floatFromInt(sample_count);
    return .{ A / samples_f, B / samples_f };
}

fn hammersley(i: u32, n: u32) [2]f32 {
    return .{ @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n)), radicalInverse(i) };
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
