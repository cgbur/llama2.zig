const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const assert = std.debug.assert;

const DEFAULT_VECTOR_WIDTH: usize = 8;

comptime {
    // TODO: seems to not have any effect
    @setFloatMode(std.builtin.FloatMode.Optimized);
}
//
/// Configuration for the model that can be read from the file. Extern and i32
/// to support the ints from python.
const ConfigReader = extern struct {
    const Self = @This();
    dim: i32, // transformer dimension
    hidden_dim: i32, // for ffn layers
    n_layers: i32, // number of layers
    n_heads: i32, // number of query heads
    n_kv_heads: i32, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    seq_len: i32, // max sequence length

    fn config(self: Self) Config {
        return Config{
            .dim = @intCast(self.dim),
            .hidden_dim = @intCast(self.hidden_dim),
            .n_layers = @intCast(self.n_layers),
            .n_heads = @intCast(self.n_heads),
            .n_kv_heads = @intCast(self.n_kv_heads),
            .vocab_size = @intCast(self.vocab_size),
            .seq_len = @intCast(self.seq_len),
        };
    }
};

/// Actual config that is used with the values as usize for ease of use.
const Config = struct {
    dim: usize, // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize, // number of layers
    n_heads: usize, // number of query heads
    n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: usize, // vocabulary size, usually 256 (byte-level)
    seq_len: usize, // max sequence length
};

/// Weights for the model held as f32 manypointers. Need to look into if slices
/// can be used for this easily.
const Weights = struct {
    token_embedding_table: [*]f32, // (vocab_size, dim)
    rms_att_weight: [*]f32, // (layer, dim) rmsnorm weights
    rms_ffn_weight: [*]f32, // (layer, dim)
    wq: [*]f32, // (layer, dim, dim)
    wk: [*]f32, // (layer, dim, dim)
    wv: [*]f32, // (layer, dim, dim)
    wo: [*]f32, // (layer, dim, dim)
    // weights for ffn
    w1: [*]f32, // (layer, hidden_dim, dim)
    w2: [*]f32, // (layer, dim, hidden_dim)
    w3: [*]f32, // (layer, hidden_dim, dim)
    rms_final_weight: [*]f32, // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: [*]f32, // (seq_len, dim/2)
    freq_cis_imag: [*]f32, // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    wcls: [*]f32, // (vocab_size, dim)

    fn init(config: *const Config, data: []u8, shared_weights: bool) Weights {
        const vocab_size: usize = config.vocab_size;
        const dim: usize = config.dim;
        const hidden_dim: usize = config.hidden_dim;
        const n_layers: usize = config.n_layers;
        const n_heads: usize = config.n_heads;
        const seq_len: usize = config.seq_len;

        var weights: Weights = undefined;

        var ptr: [*]f32 = @alignCast(@ptrCast(data));
        weights.token_embedding_table = ptr;
        ptr += vocab_size * dim;
        weights.rms_att_weight = ptr;
        ptr += n_layers * dim;
        weights.wq = ptr;
        ptr += n_layers * dim * dim;
        weights.wk = ptr;
        ptr += n_layers * dim * dim;
        weights.wv = ptr;
        ptr += n_layers * dim * dim;
        weights.wo = ptr;
        ptr += n_layers * dim * dim;
        weights.rms_ffn_weight = ptr;
        ptr += n_layers * dim;
        weights.w1 = ptr;
        ptr += n_layers * dim * hidden_dim;
        weights.w2 = ptr;
        ptr += n_layers * hidden_dim * dim;
        weights.w3 = ptr;
        ptr += n_layers * dim * hidden_dim;
        weights.rms_final_weight = ptr;
        ptr += dim;
        weights.freq_cis_real = ptr;
        var head_size: usize = dim / n_heads;
        ptr += seq_len * head_size / 2;
        weights.freq_cis_imag = ptr;
        ptr += seq_len * head_size / 2;
        weights.wcls = if (shared_weights) weights.token_embedding_table else ptr;

        return weights;
    }
};

/// The state of the model while running
const RunState = struct {
    const Self = @This();

    x: []f32, // activation at current time stamp (dim,)
    xb: []f32, // same, but inside a residual branch (dim,)
    xb2: []f32, // an additional buffer just for convenience (dim,)
    hb: []f32, // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: []f32, // buffer for hidden dimension in the ffn (hidden_dim,)
    q: []f32, // query (dim,)
    k: []f32, // key (dim,)
    v: []f32, // value (dim,)
    att: []f32, // buffer for scores/attention values (n_heads, seq_len)
    logits: []f32, // output logits
    // kv cache
    key_cache: []f32, // (layer, seq_len, dim)
    value_cache: []f32, // (layer, seq_len, dim)

    fn init(allocator: Allocator, config: *const Config) !Self {
        return Self{
            .x = try allocator.alloc(f32, config.dim),
            .xb = try allocator.alloc(f32, config.dim),
            .xb2 = try allocator.alloc(f32, config.dim),
            .hb = try allocator.alloc(f32, config.hidden_dim),
            .hb2 = try allocator.alloc(f32, config.hidden_dim),
            .q = try allocator.alloc(f32, config.dim),
            .k = try allocator.alloc(f32, config.dim),
            .v = try allocator.alloc(f32, config.dim),
            .att = try allocator.alloc(f32, config.n_heads * config.seq_len),
            .logits = try allocator.alloc(f32, config.vocab_size),
            .key_cache = try allocator.alloc(f32, config.n_layers * config.seq_len * config.dim),
            .value_cache = try allocator.alloc(f32, config.n_layers * config.seq_len * config.dim),
        };
    }

    fn deinit(self: *Self, allocator: Allocator) void {
        allocator.free(self.x);
        allocator.free(self.xb);
        allocator.free(self.xb2);
        allocator.free(self.hb);
        allocator.free(self.hb2);
        allocator.free(self.q);
        allocator.free(self.k);
        allocator.free(self.v);
        allocator.free(self.att);
        allocator.free(self.logits);
        allocator.free(self.key_cache);
        allocator.free(self.value_cache);
        self.* = undefined;
    }
};

/// A (vocab_size, token_len) array of tokens where each token is a string of
/// bytes (i.e. a string of u8s) that is variable length. The token_len for
/// each token is variable but the total length of the array is fixed.
const Tokens = struct {
    const Self = @This();

    tokens: [][]u8,

    fn init(reader: anytype, allocator: Allocator, vocab_size: usize) !Tokens {
        var tokens: Tokens = undefined;
        tokens.tokens = try allocator.alloc([]u8, vocab_size);

        for (0..vocab_size) |i| {
            const token_len = try reader.readInt(u32, std.builtin.Endian.Little);
            tokens.tokens[i] = try allocator.alloc(u8, token_len);
            const read_amt = try reader.read(tokens.tokens[i]);
            if (read_amt != token_len) {
                return error.UnexpectedEof;
            }
        }

        return tokens;
    }

    fn deinit(self: *Self, allocator: Allocator) void {
        for (self.tokens) |token| {
            allocator.free(token);
        }
        allocator.free(self.tokens);
        self.* = undefined;
    }
};

fn transformer(token: usize, pos: usize, config: *const Config, s: *RunState, w: *const Weights) void {
    // convenience variables
    const dim = config.dim;
    const hidden_dim = config.hidden_dim;
    const head_size = dim / config.n_heads;
    var x = s.x;

    // copy the token embedding into x
    const embedding_row = w.token_embedding_table[token * dim ..][0..dim];
    @memcpy(x, embedding_row);

    // pluck out the "pos" row of the freq_cis real and imaginary parts
    const freq_cis_real_row = w.freq_cis_real[pos * head_size / 2 ..][0 .. head_size / 2];
    const freq_cis_imag_row = w.freq_cis_imag[pos * head_size / 2 ..][0 .. head_size / 2];

    // forward all the layers
    for (0..config.n_layers) |l| {
        // attention rmsnorm
        rmsnorm(s.xb, x, w.rms_att_weight[l * dim ..][0..dim]);

        // qkv
        matmul(s.q, s.xb, w.wq[l * dim * dim ..][0 .. dim * dim]);
        matmul(s.k, s.xb, w.wk[l * dim * dim ..][0 .. dim * dim]);
        matmul(s.v, s.xb, w.wv[l * dim * dim ..][0 .. dim * dim]);

        // apply RoPE rotation to the q and k vectors for each head
        for (0..config.n_heads) |h| {
            // get the q and k vectors for this head
            const q = s.q[h * head_size ..][0..head_size];
            const k = s.k[h * head_size ..][0..head_size];
            // apply the rotation
            var i: usize = 0;
            while (i < head_size) : (i += 2) {
                const q0 = q[i];
                const q1 = q[i + 1];
                const k0 = k[i];
                const k1 = k[i + 1];
                const fcr = freq_cis_real_row[i / 2];
                const fci = freq_cis_imag_row[i / 2];
                q[i] = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;
                k[i] = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // save key,value at the current timestep to our kv cache
        const loff = l * config.seq_len * dim; // kv cache offset
        const key_cache_row = s.key_cache[loff + pos * dim ..][0..dim];
        const value_cache_row = s.value_cache[loff + pos * dim ..][0..dim];
        @memcpy(key_cache_row, s.k);
        @memcpy(value_cache_row, s.v);

        // attention
        // TODO: parallelize this loop
        for (0..config.n_heads) |h| {
            // get the query vector for this head
            const q = s.q[h * head_size ..][0..head_size];
            // attention scores
            const att = s.att[h * config.seq_len ..][0..config.seq_len];
            // iterate over the timesteps, including the current one
            for (0..pos + 1) |t| {
                // get the key for this timestep
                const k = s.key_cache[loff + t * dim + h * head_size ..][0..head_size];
                // attn score as the dot of q and k
                var score: f32 = vector_dot_product(DEFAULT_VECTOR_WIDTH, q, k);
                score /= std.math.sqrt(@as(f32, @floatFromInt(head_size)));
                // save the score
                att[t] = score;
            }

            // softmax the scores to get the attention weights for 0..pos inclusive
            softmax(att[0 .. pos + 1]);

            // weighted sum of the value vectors store back into xb
            const xb = s.xb[h * head_size ..][0..head_size];
            @memset(xb, 0);
            for (0..pos + 1) |t| {
                // get the value vec for this head and timestep
                const v = s.value_cache[loff + t * dim + h * head_size ..][0..head_size];
                // get the attention weight for this timestep
                const a = att[t];
                // accumulate the weighted value vector into xb
                vector_weighted_sum(DEFAULT_VECTOR_WIDTH, xb, v, a);
            }
        }

        // final matmul to get the output of attention
        matmul(s.xb2, s.xb, w.wo[l * dim * dim ..][0 .. dim * dim]);

        // residual connection back into x
        accum(x, s.xb2);

        // ffn rsnorm
        rmsnorm(s.xb, x, w.rms_ffn_weight[l * dim ..][0..dim]);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s.hb, s.xb, w.w1[l * dim * hidden_dim ..][0 .. dim * hidden_dim]);
        matmul(s.hb2, s.xb, w.w3[l * dim * hidden_dim ..][0 .. dim * hidden_dim]);

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for (0..hidden_dim) |i| {
            s.hb[i] = s.hb[i] * (1.0 / (1.0 + std.math.exp(-s.hb[i])));
        }

        // elementwise multiply with w3(x)
        for (0..hidden_dim) |i| {
            s.hb[i] = s.hb[i] * s.hb2[i];
        }

        // final matmul to get the output of FFN
        matmul(s.xb, s.hb, w.w2[l * dim * hidden_dim ..][0 .. hidden_dim * dim]);

        // residual connection
        accum(x, s.xb);
    }

    // final rmsnorm
    rmsnorm(x, x, w.rms_final_weight[0..dim]);

    // classify into logits
    matmul(s.logits, x, w.wcls[0 .. dim * config.vocab_size]);
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

/// W (d,n) @ x (n,) -> xout (d,)
///
/// This is a SIMD matrix multiplication function implementation. Matrices
/// dimensions are inferred from the lengths of the slices. xout must have same
/// length as the number of rows in W. x must have same length as the number of
/// columns in W. The layout of W is row-major.
///
///                  W
/// +---------+     +---------+     +---------+
/// |         |     |         |     |         |
/// |   d x n |  @  |   n x 1 |  =  |   d x 1 |
/// |         |     |         |     |         |
/// +---------+     +---------+     +---------+
///     W                x             xout
///
fn matmul(xout: []f32, x: []const f32, w: []const f32) void {
    // This one function accounts for ~90% of the total runtime.
    @setFloatMode(std.builtin.FloatMode.Optimized);
    const d = xout.len;
    const n = x.len;
    assert(w.len == n * d);
    assert(w.len > 0);

    for (0..d) |i| {
        const wrow = w[i * n ..][0..n]; // row i of W
        xout[i] = vector_dot_product(DEFAULT_VECTOR_WIDTH, wrow, x);
    }
}

/// Computes the vector addition of two vectors and then accumulates the result
/// into a scalar. Handles the case where the vector length is not a multiple
/// of the SIMD vector width.
fn vector_dot_product(comptime vector_width: usize, x: []const f32, y: []const f32) f32 {
    assert(x.len == y.len);
    const vec_len = x.len / vector_width; // num of SIMD vectors
    const vec_rem = x.len % vector_width; // num of f32 in the last SIMD vector

    // do the bulk of the work with SIMD
    var sum: @Vector(vector_width, f32) = @splat(0.0);
    var offset: usize = 0;
    for (0..vec_len) |_| {
        const xvec: @Vector(vector_width, f32) = x[offset..][0..vector_width].*;
        const yvec: @Vector(vector_width, f32) = y[offset..][0..vector_width].*;
        sum += xvec * yvec;
        offset += vector_width;
    }

    // handle the last few elements with normal scalar ops
    var sum_rem: f32 = 0.0;
    for (0..vec_rem) |i| {
        sum_rem += x[offset + i] * y[offset + i];
    }

    // reduce the SIMD vector to a scalar
    return @reduce(.Add, sum) + sum_rem;
}

/// Performs a weighted vector sum operation using SIMD for efficiency.
/// The operation performed is xout = xout + x * y where x is a vector and y is a scalar.
fn vector_weighted_sum(comptime vector_width: usize, xout: []f32, x: []const f32, y: f32) void {
    assert(xout.len == x.len);
    const vec_len = x.len / vector_width; // num of SIMD vectors
    const vec_rem = x.len % vector_width; // num of f32 in the last SIMD vector

    // do the bulk of the work with SIMD
    var offset: usize = 0;
    const yvector: @Vector(vector_width, f32) = @splat(y);
    for (0..vec_len) |_| {
        var xoutvec: @Vector(vector_width, f32) = xout[offset..][0..vector_width].*;
        const xvec: @Vector(vector_width, f32) = x[offset..][0..vector_width].*;
        xoutvec += xvec * yvector;
        xout[offset..][0..vector_width].* = xoutvec;
        offset += vector_width;
    }

    // handle the last few elements with normal scalar operations
    for (0..vec_rem) |i| {
        xout[offset + i] += x[offset + i] * y;
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

fn accum(a: []f32, b: []f32) void {
    assert(a.len == b.len);
    for (0..a.len) |i| {
        a[i] += b[i];
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

fn sample(x: []f32) usize {
    assert(x.len > 0);
    var rng = std.rand.DefaultPrng.init(0);
    var r = rng.random().float(f32);

    var cdf: f32 = 0.0;
    for (x, 0..) |*val, i| {
        cdf += val.*;
        if (r < cdf) {
            return i;
        }
    }
    return x.len - 1;
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const args = try std.process.argsAlloc(allocator);
    if (args.len < 2) {
        std.debug.print("usage: llama2 <checkpoint.bin> [temperature=0.9]\n", .{});
        return;
    }

    // grab the checkpoint path
    const bin_path = args[1];

    const DEFAULT_TEMPERATURE = 0.9;
    var temperature: f32 = if (args.len > 2)
        std.fmt.parseFloat(f32, args[2]) catch DEFAULT_TEMPERATURE
    else
        DEFAULT_TEMPERATURE;
    temperature = std.math.clamp(temperature, 0.0, 1.0);

    // read the config from the checkpoint
    var checkpoint = try std.fs.cwd().openFile(bin_path, .{}); // close this by hand
    var config_read: ConfigReader = try checkpoint.reader().readStruct(ConfigReader);
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    const shared_weights: bool = config_read.vocab_size > 0;
    config_read.vocab_size = try std.math.absInt(config_read.vocab_size);
    const file_size = (try checkpoint.stat()).size;
    checkpoint.close();
    const config = config_read.config(); // convert to usize version

    std.debug.print("config: {any}\n", .{config});
    std.debug.print("shared weights: {any}\n", .{shared_weights});
    std.debug.print("temperature: {d}\n", .{temperature});
    std.debug.print("\n", .{});

    // mmap the checkpoint to directly map the weights
    const mapped_checkpoint = try (std.fs.cwd().openFile(bin_path, .{}));
    defer mapped_checkpoint.close();
    const data: []align(mem.page_size) u8 = try std.os.mmap(null, file_size, std.os.linux.PROT.READ, std.os.linux.MAP.PRIVATE, mapped_checkpoint.handle, 0);
    defer std.os.munmap(data);
    const weights = Weights.init(&config, data[@sizeOf(ConfigReader)..], shared_weights);

    // load the tokens for the model
    var tokens = tokenblk: {
        var token_file = try std.fs.cwd().openFile("tokenizer.bin", .{});
        defer token_file.close();
        var buf_reader = std.io.bufferedReader(token_file.reader());
        const tokens = try Tokens.init(buf_reader.reader(), allocator, config.vocab_size);
        break :tokenblk tokens;
    };
    defer tokens.deinit(allocator);

    // initialize the run state for inference
    var state = try RunState.init(allocator, &config);
    defer state.deinit(allocator);

    var stdout = std.io.getStdOut().writer();

    var next: usize = undefined; // the next token as predicted by the model
    var token: usize = 1; // 1 = <BOS> for llama2
    var timer: ?std.time.Timer = null;

    // for now just do seq len steps
    for (0..config.seq_len) |pos| {
        transformer(token, pos, &config, &state, &weights);

        if (temperature == 0.0) {
            next = argmax(state.logits);
        } else {
            // apply the temperature to the logits
            for (state.logits) |*val| val.* /= temperature;
            // apply softmax to the logits to get the probabilities for next token
            softmax(state.logits);
            next = sample(state.logits);
        }

        // print the token, don't bother with the white space hack for now
        var token_str = tokens.tokens[next];
        try stdout.print("{s}", .{token_str});
        token = next;

        // if timer is null, start it
        if (timer == null) {
            timer = try std.time.Timer.start();
        }
    }
    const time = timer.?.read();
    const tokens_per_ms = @as(f64, @floatFromInt(config.seq_len - 1)) / @as(f64, @floatFromInt(time / std.time.ns_per_ms));
    const tokens_per_sec = tokens_per_ms * 1000.0;

    // print tokens per second
    std.debug.print("\n\n{d} tokens per second\n", .{tokens_per_sec});
}

test "matrix_multiplies" {
    var w = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    var x = [_]f32{ 1.0, 2.0, 3.0 };
    var xout = [_]f32{ 0.0, 0.0, 0.0 };

    matmul(&xout, &x, &w);
    try std.testing.expect(xout[0] == 1.0 + 4.0 + 9.0);
    try std.testing.expect(xout[1] == 4.0 + 10.0 + 18.0);
    try std.testing.expect(xout[2] == 7.0 + 16.0 + 27.0);
}

test "vector_length_less_than_width_case" {
    var w = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
    var x = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var xout = [_]f32{ 0, 0 };

    matmul(&xout, &x, &w);

    var expectedResult = [_]f32{ 0, 0 };
    for (0..2) |i| {
        for (0..12) |j| {
            expectedResult[i] += w[i * 12 + j] * x[j];
        }
        try std.testing.expect(xout[i] == expectedResult[i]);
    }
}

test "vector_weighted_sum_length_less_than_width_case" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    var y: f32 = 3.0;
    var xout = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };

    vector_weighted_sum(8, &xout, &x, y);
    for (0..xout.len) |i| {
        var expected = (x[i] * y) + x[i];
        try std.testing.expect((xout[i] - expected) < 0.0001);
    }
}
