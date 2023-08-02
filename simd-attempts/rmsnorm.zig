const std = @import("std");
const assert = std.debug.assert;
const DEFAULT_VECTOR_WIDTH = 8;

fn rmsnorm_vectorized(o: []f32, x: []f32, w: []f32) void {
    assert(o.len == x.len);
    assert(o.len == w.len);

    const dim = o.len;
    const vector_width = DEFAULT_VECTOR_WIDTH;
    const vec_len = dim / vector_width; // num of SIMD vectors
    const vec_rem = dim % vector_width; // num of f32 in the last SIMD vector

    // sum of squares
    var vec_sum: @Vector(vector_width, f32) = @splat(0.0);
    var offset: usize = 0;
    for (0..vec_len) |_| {
        const xvec: @Vector(vector_width, f32) = x[offset..][0..vector_width].*;
        vec_sum += xvec * xvec;
        offset += vector_width;
    }
    var sum_rem: f32 = 0.0;
    for (0..vec_rem) |i| {
        sum_rem += x[offset + i] * x[offset + i];
    }
    var sum = @reduce(.Add, vec_sum) + sum_rem;
    sum /= @floatFromInt(dim);
    sum += 1e-5;
    sum = 1.0 / std.math.sqrt(sum);
    const sum_vec: @Vector(vector_width, f32) = @splat(sum);

    // normalize and scale
    offset = 0;
    for (0..vec_len) |_| {
        const xvec: @Vector(vector_width, f32) = x[offset..][0..vector_width].*;
        const wvec: @Vector(vector_width, f32) = w[offset..][0..vector_width].*;
        o[offset..][0..vector_width].* = xvec * sum_vec * wvec;
        offset += vector_width;
    }
    for (0..vec_rem) |i| {
        o[offset + i] = x[offset + i] * sum * w[offset + i];
    }
}

fn rmsnorm(o: []f32, x: []f32, w: []f32) void {
    assert(o.len == x.len);
    assert(o.len == w.len);

    // sum of squares
    var sum: f32 = 0.0;
    for (x) |val| {
        sum += val * val;
    }
    sum /= @floatFromInt(x.len);
    sum += 1e-5;
    sum = 1.0 / std.math.sqrt(sum);

    // normalize and scale
    for (0..o.len) |i| {
        o[i] = x[i] * sum * w[i];
    }
}

pub fn main() !void {
    // generate 1mb of f32 random data
    var rng = std.rand.DefaultPrng.init(0);
    var r = rng.random();
    const size: usize = 2988;
    const num_runs: usize = 100000;
    var o: [size]f32 = undefined;
    var x: [size]f32 = undefined;
    var w: [size]f32 = undefined;

    for (0..size) |i| {
        o[i] = r.float(f32);
        x[i] = r.float(f32);
        w[i] = r.float(f32);
    }

    var timer = try std.time.Timer.start();
    timer.reset();
    for (0..num_runs) |_| {
        rmsnorm(&o, &x, &w);
    }
    const softmax_ns: u64 = timer.read();

    timer.reset();
    for (0..num_runs) |_| {
        rmsnorm_vectorized(&o, &x, &w);
    }
    const softmax_vectored_ns: u64 = timer.read();

    std.debug.print("{:15} ns/rmsnorm\n", .{softmax_ns / num_runs});
    std.debug.print("{:15} ns/rmsnorm_vectored\n", .{softmax_vectored_ns / num_runs});
}
