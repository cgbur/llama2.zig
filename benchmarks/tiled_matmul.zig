const std = @import("std");
const assert = std.debug.assert;
const DEFAULT_VECTOR_WIDTH = 8;

fn matmul_mul_tiled(xout: []f32, x: []const f32, w: []const f32) void {
    // This one function accounts for ~90% of the total runtime.
    const d = xout.len;
    const n = x.len;
    assert(w.len == n * d);
    assert(w.len > 0);

    // variables to unroll the loop so we can be more cache friendly with x
    const unroll_factor = 2;
    const unroll_len = d / unroll_factor;
    const unroll_rem = d % unroll_factor;

    // variables to handle the SIMD vectorization
    const vector_width = DEFAULT_VECTOR_WIDTH;
    const vec_len = x.len / vector_width; // num of SIMD vectors
    const vec_rem = x.len % vector_width; // num of f32 in the last SIMD vector

    var wrows: [unroll_factor][]const f32 = undefined;
    var sums: [unroll_factor]@Vector(vector_width, f32) = [1]@Vector(vector_width, f32){@splat(0.0)} ** unroll_factor;
    var sums_rem: [unroll_factor]f32 = [1]f32{0.0} ** unroll_factor;

    var row_offset: usize = 0;
    for (0..unroll_len) |_| {
        inline for (0..unroll_factor) |j| {
            wrows[j] = w[(row_offset * unroll_factor + j) * n ..][0..n];
            @prefetch(wrows[j].ptr, .{ .locality = 0 });
            // reset sums
            sums[j] = @splat(0.0);
            sums_rem[j] = 0.0;
        }

        // do the bulk of the work with SIMD
        var offset: usize = 0;
        for (0..vec_len) |_| {
            const xvec: @Vector(vector_width, f32) = x[offset..][0..vector_width].*;
            inline for (0..unroll_factor) |j| {
                const wvec: @Vector(vector_width, f32) = wrows[j][offset..][0..vector_width].*;
                sums[j] += xvec * wvec;
            }
            offset += vector_width;
        }

        // handle the last few elements with normal scalar ops
        for (0..vec_rem) |a| {
            inline for (0..unroll_factor) |j| {
                sums_rem[j] += x[offset + a] * wrows[j][offset + a];
            }
        }

        // reduce SIMD vector to scalar
        inline for (0..unroll_factor) |j| {
            xout[row_offset * unroll_factor + j] = @reduce(.Add, sums[j]) + sums_rem[j];
        }

        row_offset += 1;
    }

    // handle the last rows of the matrix
    for (0..unroll_rem) |_| {
        // just call vector_dot_product for the last few rows
        const wrow: []const f32 = w[(row_offset * unroll_factor) * n ..][0..n];
        _ = wrow;
        // xout[row_offset * unroll_factor] = vector_dot_product(x, wrow);
        row_offset += 1;
    }
}
