const std = @import("std");
const sokol = @import("sokol");
const Build = std.Build;

/// Build-time quality settings for the embedded split-sum BRDF lookup table.
///
/// Zig dependency arguments are intentionally flat, so a consuming project
/// supplies these as `.brdf_lut_resolution` and `.brdf_lut_samples` to
/// `b.dependency("Zupra", ...)`. Keeping them together here documents the
/// coherent setting group and gives this package one validation boundary.
pub const BrdfLutBuildSettings = struct {
    /// Width and height of the square RG16F LUT. The shipped 512x512 asset is
    /// used directly for the default, so normal builds never wait for a bake.
    resolution: u32 = 512,
    /// GGX samples evaluated for each LUT texel when a custom LUT is baked.
    sample_count: u32 = 4096,

    fn isDefault(self: BrdfLutBuildSettings) bool {
        return self.resolution == 512 and self.sample_count == 4096;
    }
};

pub fn build(b: *Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    // A package dependency should expose its framework module, never compile
    // the playground merely because the consuming project is a Debug build.
    const build_example = b.option(bool, "example", "Build the playground executable (default: Debug only for Zupra itself)") orelse (b.dep_prefix.len == 0 and optimize == .Debug);
    // Generated shader bindings are checked in. Regenerating them is a source
    // tree operation for framework development, not something an application
    // importing Zupra should attempt in its dependency cache.
    const regenerate_shaders = b.option(bool, "regenerate-shaders", "Regenerate checked-in shader bindings (default: only for Zupra itself)") orelse (b.dep_prefix.len == 0);
    const brdf_lut_settings = try readBrdfLutBuildSettings(b);

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

    const shaders_mod = b.createModule(.{
        .root_source_file = b.path("shaders/shaders.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "sokol", .module = dep_sokol.module("sokol") }},
    });

    const brdf_lut_mod = addEmbeddedBrdfLutModule(b, target, optimize, brdf_lut_settings);

    // Public framework module for downstream projects:
    //   exe.root_module.addImport("zupra", zupra_dep.module("zupra"));
    const root_mod = b.addModule("zupra", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sokol", .module = dep_sokol.module("sokol") },
            .{ .name = "zstbi", .module = zstbi.module("root") },
            .{ .name = "zmesh", .module = zmesh.module("root") },
            .{ .name = "zmath", .module = zmath.module("root") },
            .{ .name = "shaders", .module = shaders_mod },
            .{ .name = "brdf_lut", .module = brdf_lut_mod },
        },
    });

    const debug_views = b.option(bool, "debug-views", "Compile in debug visualisations (AO buffer, cluster heatmap, etc.)") orelse (optimize == .Debug);
    const build_opts = b.addOptions();
    build_opts.addOption(bool, "debug_views", debug_views);
    root_mod.addOptions("build_options", build_opts);

    var compile_step: *std.Build.Step.Compile = undefined;

    const example_mod = b.createModule(.{
        .root_source_file = b.path("examples/playground.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zupra", .module = root_mod },
            .{ .name = "sokol", .module = dep_sokol.module("sokol") },
            .{ .name = "shaders", .module = shaders_mod },
        },
    });

    const root_source = if (build_example) example_mod else root_mod;

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
        if (build_example) {
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
        .flags = &.{"-fno-sanitize=undefined"},
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
        // Keep the public module self-contained when an application imports it
        // instead of using this package's own executable target.
        root_mod.link_libc = true;
    }

    // The linkage belongs to the public module, not only this package's own
    // artifact. That lets a consuming executable inherit zmesh correctly.
    root_mod.linkLibrary(zmesh.artifact("zmesh"));

    if (regenerate_shaders) {
        try compileShaders(b, compile_step, dep_sokol, "shaders");
        if (build_example) {
            try compileShaders(b, compile_step, dep_sokol, "examples/assets/shaders");
        }
    }
    // Expose include shaders to the outside.
    const shader_includes = b.addNamedWriteFiles("shader-includes");
    _ = shader_includes.addCopyFile(b.path("shaders/material_surface.glsl.inc"), "material_surface.glsl.inc");
    _ = shader_includes.addCopyFile(b.path("shaders/material_alpha.glsl.inc"), "material_alpha.glsl.inc");
    _ = shader_includes.addCopyFile(b.path("shaders/pbr_lib.glsl.inc"), "pbr_lib.glsl.inc");

    // Keep the framework's pure renderer/asset tests runnable with the same
    // module graph as the application. Direct `zig test src/...` invocations
    // cannot resolve package imports such as sokol, zmesh and shaders.
    const unit_tests = b.addTest(.{ .root_module = root_mod });
    unit_tests.step.dependOn(&compile_step.step);
    const run_unit_tests = b.addRunArtifact(unit_tests);
    b.step("test", "Run Zupra unit tests").dependOn(&run_unit_tests.step);
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

// --- shaders ---

fn compileShaders(b: *Build, compile_step: *std.Build.Step.Compile, dep_sokol: *Build.Dependency, shader_dir_path: []const u8) !void {
    const dep_shdc = dep_sokol.builder.dependency("shdc", .{});
    const io = b.graph.io;
    // A dependency's build script runs with the consuming application's working
    // directory. Resolve shader folders from this package instead.
    var dir = try b.build_root.handle.openDir(io, shader_dir_path, .{ .iterate = true });
    defer dir.close(io);

    // Pass 1: collect every .inc include (shared shader libs).
    var includes: std.ArrayList([]const u8) = .empty;
    {
        var iter = dir.iterate();
        while (try iter.next(io)) |entry| {
            if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".inc")) {
                try includes.append(b.allocator, b.pathJoin(&.{ shader_dir_path, entry.name }));
            }
        }
    }

    // Pass 2: compile each .glsl and register the includes on its Run step.
    var iter = dir.iterate();
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

            // shdc_step is the UpdateSourceFiles copy step; the actual shdc Run
            // step is its single dependency. Register each .inc as a file input
            // on that Run step so its cache key includes the include contents.
            for (shdc_step.dependencies.items) |dep_step| {
                if (dep_step.cast(std.Build.Step.Run)) |run| {
                    for (includes.items) |inc_path| {
                        run.addFileInput(b.path(inc_path));
                    }
                }
            }

            compile_step.step.dependOn(shdc_step);
        }
    }
}

pub const ShaderOptions = struct {
    source: std.Build.LazyPath,
    /// Output name, e.g. "hologram.glsl.zig"
    name: []const u8,
};

/// Compile a user shader against Zupra's shader includes. Returns the generated
/// .glsl.zig LazyPath, ready to hand to b.createModule.
pub fn addShader(
    b: *std.Build,
    zupra_dep: *std.Build.Dependency,
    shdc: *std.Build.Step.Run, // your existing shdc run step factory
    opts: ShaderOptions,
) std.Build.LazyPath {
    const stage = b.addWriteFiles();
    // Zupra's includes first, then the user's shader alongside them.
    _ = stage.addCopyDirectory(zupra_dep.namedLazyPath("shader-includes"), "", .{});
    const staged_src = stage.addCopyFile(opts.source, b.fmt("{s}", .{std.fs.path.basename(opts.source.getDisplayName())}));

    const run = b.addRunArtifact(shdc);
    run.addArg("--input");
    run.addFileArg(staged_src);
    run.addArg("--output");
    const out = run.addOutputFileArg(opts.name);
    run.addArgs(&.{ "--slang", "hlsl5:glsl430:metal_macos:wgsl", "--format", "sokol_zig" });
    return out;
}

// --- embedded BRDF LUT ---

fn readBrdfLutBuildSettings(b: *Build) !BrdfLutBuildSettings {
    const settings = BrdfLutBuildSettings{
        .resolution = b.option(u32, "brdf_lut_resolution", "Embedded BRDF LUT width/height (default: 512)") orelse 512,
        .sample_count = b.option(u32, "brdf_lut_samples", "GGX samples per BRDF LUT texel when baking a custom LUT (default: 4096)") orelse 4096,
    };

    // Non-power-of-two dimensions are valid GPU texture sizes, but zero, very
    // large images, and empty integration loops are invariably configuration
    // errors. The cap also prevents an accidental build option from allocating
    // an impractical amount of memory in the host-side baker.
    if (settings.resolution == 0 or settings.resolution > 4096) {
        std.log.err("brdf_lut_resolution must be in 1..4096, got {d}", .{settings.resolution});
        return error.InvalidBrdfLutResolution;
    }
    if (settings.sample_count == 0 or settings.sample_count > 65_536) {
        std.log.err("brdf_lut_samples must be in 1..65536, got {d}", .{settings.sample_count});
        return error.InvalidBrdfLutSampleCount;
    }
    return settings;
}

/// Make an internal module whose sibling LUT is embedded at compile time.
///
/// The default is the checked-in high-quality asset, keeping normal builds
/// instant. A non-default setting runs the host-side baker into Zig's cache;
/// it never overwrites the package source tree and its output is keyed by the
/// selected dimensions/sample count. In both cases the final executable owns
/// the bytes, so it works from any working directory and on every target.
fn addEmbeddedBrdfLutModule(
    b: *Build,
    target: Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    settings: BrdfLutBuildSettings,
) *Build.Module {
    const lut = if (settings.isDefault())
        b.path("resources/brdf.lut")
    else
        bakeCustomBrdfLut(b, settings);

    // @embedFile resolves relative to the source file declaring it. Stage both
    // generated source and LUT into one cache directory so that relationship is
    // explicit in the build graph rather than depending on the process CWD.
    const stage = b.addWriteFiles();
    _ = stage.addCopyFile(lut, "brdf.lut");
    const source = stage.add("brdf_lut.zig", b.fmt(
        \\pub const resolution: u32 = {d};
        \\pub const sample_count: u32 = {d};
        \\pub const bytes: []const u8 = @embedFile("brdf.lut");
    , .{ settings.resolution, settings.sample_count }));

    return b.createModule(.{
        .root_source_file = source,
        .target = target,
        .optimize = optimize,
    });
}

fn bakeCustomBrdfLut(b: *Build, settings: BrdfLutBuildSettings) Build.LazyPath {
    // This is a build-time executable and must run on the host even when the
    // game itself is being cross-compiled for web, Android, or iOS.
    const baker = b.addExecutable(.{
        .name = "zupra_brdf_lut_baker",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/brdflut_backer.zig"),
            .target = b.graph.host,
            .optimize = .ReleaseFast,
        }),
    });
    const run = b.addRunArtifact(baker);
    run.addArgs(&.{
        "--resolution",
        b.fmt("{d}", .{settings.resolution}),
        "--samples",
        b.fmt("{d}", .{settings.sample_count}),
        "--output",
    });
    return run.addOutputFileArg("brdf.lut");
}
