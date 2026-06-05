const std = @import("std");
const sokol = @import("sokol");
const Build = std.Build;

pub fn build(b: *Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // == Dependencies ==

    const dep_sokol = b.dependency("sokol", .{
        .target = target,
        .optimize = optimize,
    });
    const zstbi = b.dependency("zstbi", .{
        .target = target,
        .optimize = optimize,
    });
    const zmesh = b.dependency("zmesh", .{
        .target = target,
        .optimize = optimize,
    });
    const zmath = b.dependency("zmath", .{
        .target = target,
        .optimize = optimize,
    });

    // == Root Module ==

    const root_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sokol", .module = dep_sokol.module("sokol") },
            .{ .name = "zstbi", .module = zstbi.module("root") },
            .{ .name = "zmesh", .module = zmesh.module("root") },
            .{ .name = "zmath", .module = zmath.module("root") },
            .{ .name = "shaders", .module = b.createModule(.{
                .root_source_file = b.path("shaders/shaders.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "sokol", .module = dep_sokol.module("sokol") }},
            }) },
        },
    });

    var compile_step: *std.Build.Step.Compile = undefined;

    const example_mod = b.createModule(.{
        .root_source_file = b.path("examples/playground.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zupra", .module = root_mod },
            .{ .name = "sokol", .module = dep_sokol.module("sokol") }, // make sokol available for examples/assets/shaders/
        },
    });

    const root_source = if (optimize == .Debug) example_mod else root_mod;

    // build for wasm (emscripten)
    if (target.result.cpu.arch.isWasm()) {
        compile_step = b.addLibrary(.{
            .name = "Zupra",
            .linkage = .static,
            .root_module = root_source,
        });
        try buildWeb(b, compile_step, root_mod, dep_sokol);
    } else if (target.result.abi.isAndroid()) { // build for android
        compile_step = b.addLibrary(.{
            .name = "Zupra",
            .linkage = .dynamic,
            .root_module = root_source,
        });

        const ndk_path = b.option([]const u8, "android_ndk", "Path to Android NDK") orelse {
            std.debug.print("error: Android NDK path required, pass -Dandroid_ndk=/path/to/ndk\n", .{});
            return error.MissingAndroidNdk;
        };

        const ndk_include = b.fmt("{s}/toolchains/llvm/prebuilt/windows-x86_64/sysroot/include", .{ndk_path});
        root_mod.addSystemIncludePath(.{ .cwd_relative = ndk_include });
        zmesh.module("root").addSystemIncludePath(.{ .cwd_relative = ndk_include });
        zmesh.artifact("zmesh").root_module.addSystemIncludePath(.{ .cwd_relative = ndk_include });
        zstbi.module("root").addSystemIncludePath(.{ .cwd_relative = ndk_include });

        b.installArtifact(compile_step);
    } else if (target.result.os.tag == .ios) { // build for ios
        compile_step = b.addLibrary(.{
            .name = "Zupra",
            .linkage = .static,
            .root_module = root_source,
        });
        b.installArtifact(compile_step);
    } else { // build native (desktop)
        if (optimize == .Debug) {
            compile_step = b.addExecutable(.{
                .name = "Zupra",
                .root_module = root_source,
            });
            try buildNative(b, compile_step);
        } else {
            compile_step = b.addLibrary(.{
                .name = "Zupra",
                .linkage = .static,
                .root_module = root_source,
            });
            if (target.result.os.tag == .windows) {
                compile_step.subsystem = .Windows;
            }
            b.installArtifact(compile_step);
        }
    }

    // add c-libs
    root_mod.addCSourceFile(.{
        .file = b.path("libs/stb_truetype/ctbtt_impl.c"),
        .flags = &.{},
    });
    root_mod.addIncludePath(b.path("libs/stb_truetype"));

    // add dependencies to include path
    if (target.result.cpu.arch.isWasm()) {
        const emsdk = dep_sokol.builder.dependency("emsdk", .{});
        const sysroot_include = emsdk.path("upstream/emscripten/cache/sysroot/include");

        root_mod.addSystemIncludePath(sysroot_include);
        compile_step.root_module.addSystemIncludePath(sysroot_include);
        zstbi.module("root").addSystemIncludePath(sysroot_include);
        zmesh.module("root").addSystemIncludePath(sysroot_include);
        zmesh.artifact("zmesh").root_module.addSystemIncludePath(sysroot_include);
    } else {
        compile_step.is_linking_libc = true;
    }

    compile_step.root_module.linkLibrary(zmesh.artifact("zmesh"));

    try compileShaders(b, compile_step, dep_sokol, "shaders");
    try compileShaders(b, compile_step, dep_sokol, "examples/assets/shaders");
    bakeBrdfLut(b, &target, compile_step);
}

fn buildNative(b: *Build, exe: *std.Build.Step.Compile) !void {
    b.installArtifact(exe);
    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    b.step("run", "Run playground").dependOn(&run.step);
}

fn buildWeb(b: *Build, lib: *std.Build.Step.Compile, root: *Build.Module, dep_sokol: *Build.Dependency) !void {
    const emsdk = dep_sokol.builder.dependency("emsdk", .{});

    const link_step = try sokol.emLinkStep(b, .{
        .lib_main = lib,
        .target = root.resolved_target.?,
        .optimize = root.optimize.?,
        .emsdk = emsdk,
        .use_webgl2 = true,
        .use_emmalloc = true,
        .use_filesystem = true,
        .shell_file_path = b.path("resources/shell.html"),
        .extra_args = &.{ "-sALLOW_MEMORY_GROWTH=1", "-sNO_FILESYSTEM=0" },
    });

    b.getInstallStep().dependOn(&link_step.step);

    const run = sokol.emRunStep(b, .{ .name = "Zupra", .emsdk = emsdk });
    run.step.dependOn(&link_step.step);
    b.step("run", "Run engine").dependOn(&run.step);
}

fn compileShaders(b: *Build, compile_step: *std.Build.Step.Compile, dep_sokol: *Build.Dependency, shader_dir_path: []const u8) !void {
    // Shader compilation
    const dep_shdc = dep_sokol.builder.dependency("shdc", .{});
    const io = b.graph.io;
    const cwd = std.Io.Dir.cwd();
    var dir = try cwd.openDir(io, shader_dir_path, .{ .iterate = true });
    defer dir.close(io);

    var iter = dir.iterate(); // in 0.16.0 iterate() may need io — see note below
    while (try iter.next(io)) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".glsl")) {
            const input_path = b.pathJoin(&.{ shader_dir_path, entry.name });
            const output_name = b.fmt("{s}.zig", .{entry.name});
            const output_path = b.pathJoin(&.{ shader_dir_path, output_name });

            const shdc_step = try sokol.shdc.createSourceFile(b, .{
                .shdc_dep = dep_shdc,
                .input = input_path,
                .output = output_path,
                .slang = .{
                    .glsl300es = true,
                    .hlsl5 = true,
                    .metal_macos = true,
                    .glsl410 = true,
                },
            });

            compile_step.step.dependOn(shdc_step);
        }
    }
}

fn bakeBrdfLut(b: *Build, target: *const Build.ResolvedTarget, compile_step: *Build.Step.Compile) void {
    // BRDF LUT bake
    const io = b.graph.io;
    const lut_path = "resources/brdf.lut";
    const needs_bake = blk: {
        std.Io.Dir.cwd().access(io, lut_path, .{}) catch {
            break :blk true;
        };
        break :blk false;
    };

    if (needs_bake) {
        const gen_brdf_exe = b.addExecutable(.{
            .name = "gen_brdf",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tools/brdflut_backer.zig"),
                .target = target.*,
                .optimize = .ReleaseFast,
            }),
        });
        const gen_brdf_run = b.addRunArtifact(gen_brdf_exe);
        gen_brdf_run.setCwd(b.path("."));
        compile_step.step.dependOn(&gen_brdf_run.step);
    }
}
