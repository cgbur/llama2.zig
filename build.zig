const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "llama2.zig",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const disable_strip = b.option(bool, "nostrip", "Disable stripping binaries, default is to strip release binaries") orelse false;

    const is_debug = optimize == .Debug;
    if (!is_debug and !disable_strip) {
        exe.root_module.strip = true;
    }

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);

    const fmt_include_paths = &.{ "src", "build.zig" };
    const do_fmt = b.addFmt(.{
        .paths = fmt_include_paths,
    });

    b.step("fmt", "Modify source files in place to have conforming formatting")
        .dependOn(&do_fmt.step);
}
