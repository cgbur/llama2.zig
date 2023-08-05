const std = @import("std");
const assert = std.debug.assert;
const DEFAULT_VECTOR_WIDTH = 8;

fn sample(x: []f32) usize {
    assert(x.len > 0);
    var rng = std.rand.DefaultPrng.init(0);
    var r = rng.random().float(f32);
    r = 0.9;

    var cdf: f32 = 0.0;
    for (x, 0..) |*val, i| {
        cdf += val.*;
        if (r < cdf) {
            return i;
        }
    }
    return x.len - 1;
}

fn softmax(x: []f32) void {
    assert(x.len > 0);
    // max of x for numerical stability
    var max: f32 = x[0];
    for (x[1..]) |val| {
        if (val > max) {
            max = val;
        }
    }
    // exp and sum
    var sum: f32 = 0.0;
    for (x) |*val| {
        val.* = std.math.exp(val.* - max);
        sum += val.*;
    }
    // normalize
    for (x) |*val| {
        val.* /= sum;
    }
}

fn argmax(x: []f32) usize {
    assert(x.len > 0);
    var max: f32 = x[0];
    var maxi: usize = 0;
    for (1..x.len) |i| {
        if (x[i] > max) {
            max = x[i];
            maxi = i;
        }
    }
    return maxi;
}
pub fn main() !void {
    // generate 1mb of f32 random data
    var rng = std.rand.DefaultPrng.init(0);
    var r = rng.random();
    const size: usize = 32_000;
    const num_runs: usize = 10000;
    var data: [size]f32 = undefined;

    for (&data) |*val| {
        val.* = r.float(f32);
    }
    data[size - 100] = 100.0;
    softmax(&data);
    var timer = try std.time.Timer.start();
    timer.reset();
    var data_copy: [size]f32 = undefined;
    for (0..num_runs) |_| {
        @memcpy(&data_copy, &data);
        var res = sample(&data_copy);
        std.mem.doNotOptimizeAway(res);
    }
    const sample_time: u64 = timer.read();

    timer.reset();
    for (0..num_runs) |_| {
        @memcpy(&data_copy, &data);
        var res = argmax(&data_copy);
        std.mem.doNotOptimizeAway(res);
    }
    const argmax_time: u64 = timer.read();

    std.debug.print("{:15} ns/sample\n", .{sample_time / num_runs});
    std.debug.print("{:15} ns/argmax\n", .{argmax_time / num_runs});
}
