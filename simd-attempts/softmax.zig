const std = @import("std");
const assert = std.debug.assert;
const DEFAULT_VECTOR_WIDTH = 8;

// was not faster than the scalar version
fn softmax_vectored(x: []f32) void {
    assert(x.len > 0);

    // determine maximum element for numerical stability
    var max: f32 = x[0];
    for (x[1..]) |val| {
        if (val > max) {
            max = val;
        }
    }

    const vector_width = DEFAULT_VECTOR_WIDTH;
    const vec_len = x.len / vector_width; // num of SIMD vectors
    const vec_rem = x.len % vector_width; // num of f32 in the last SIMD vector

    // perform e^(x - max) and accumulate sum in a SIMD-friendly manner
    var sum_vector: @Vector(vector_width, f32) = @splat(0.0);
    const max_vector: @Vector(vector_width, f32) = @splat(max);
    var offset: usize = 0;
    for (0..vec_len) |_| {
        var xvec: @Vector(vector_width, f32) = x[offset..][0..vector_width].*;
        xvec = xvec - max_vector;
        xvec = std.math.exp(xvec);
        x[offset..][0..vector_width].* = xvec;
        sum_vector += xvec;
        offset += vector_width;
    }

    // handle remaining elements with standard scalar operations
    var sum_remainder: f32 = 0.0;
    for (0..vec_rem) |i| {
        x[offset + i] = std.math.exp(x[offset + i] - max);
        sum_remainder += x[offset + i];
    }

    // calculate total sum and normalize each element
    const total_sum = @reduce(.Add, sum_vector) + sum_remainder;
    const total_sum_vector: @Vector(vector_width, f32) = @splat(total_sum);

    offset = 0;
    for (0..vec_len) |_| {
        var xvec: @Vector(vector_width, f32) = x[offset..][0..vector_width].*;
        xvec /= total_sum_vector;
        x[offset..][0..vector_width].* = xvec;
        offset += vector_width;
    }

    // normalize remaining elements with standard scalar operations
    for (0..vec_rem) |i| {
        x[offset + i] /= total_sum;
    }
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

pub fn main() !void {
    // generate 1mb of f32 random data
    var rng = std.rand.DefaultPrng.init(0);
    var r = rng.random();
    const size: usize = 30_000;
    const num_runs: usize = 10000;
    var data: [size]f32 = undefined;
    for (&data) |*val| {
        val.* = r.float(f32);
    }

    var timer = try std.time.Timer.start();
    timer.reset();
    var data_copy: [size]f32 = undefined;
    for (0..num_runs) |_| {
        @memcpy(&data_copy, &data);
        softmax(&data_copy);
    }
    const softmax_ns: u64 = timer.read();

    timer.reset();
    for (0..num_runs) |_| {
        @memcpy(&data_copy, &data);
        softmax_vectored(&data_copy);
    }
    const softmax_vectored_ns: u64 = timer.read();

    std.debug.print("{:15} ns/softmax\n", .{softmax_ns / num_runs});
    std.debug.print("{:15} ns/softmax_vectored\n", .{softmax_vectored_ns / num_runs});
}
