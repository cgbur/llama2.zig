const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const assert = std.debug.assert;
const ThreadPool = std.Thread.Pool;

const DEFAULT_VECTOR_WIDTH: usize = std.simd.suggestVectorLength(f32) orelse 4;
const simd_align = @alignOf(@Vector(DEFAULT_VECTOR_WIDTH, f32));

comptime {
    @setFloatMode(.optimized);
}

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
    // weights for matmuls (dim == n_heads * head_size)
    wq: [*]f32, // (layer, dim, n_heads * head_size)
    wk: [*]f32, // (layer, dim, n_kv_heads * head_size)
    wv: [*]f32, // (layer, dim, n_kv_heads * head_size)
    wo: [*]f32, // (layer, n_heads * head_size, dim)
    // weights for ffn
    w1: [*]f32, // (layer, hidden_dim, dim)
    w2: [*]f32, // (layer, dim, hidden_dim)
    w3: [*]f32, // (layer, hidden_dim, dim)
    rms_final_weight: [*]f32, // (dim,)
    // freq_cis for RoPE relatively positional embeddings (not used currently)
    freq_cis_real: [*]f32, // (seq_len, head_size/2)
    freq_cis_imag: [*]f32, // (seq_len, head_size/2)
    // (optional) classifier weights for the logits, on the last layer
    wcls: [*]f32, // (vocab_size, dim)

    fn init(config: *const Config, data: []u8, shared_weights: bool) Weights {
        const vocab_size: usize = config.vocab_size;
        const dim: usize = config.dim;
        const hidden_dim: usize = config.hidden_dim;
        const n_layers: usize = config.n_layers;
        const n_heads: usize = config.n_heads;
        const n_kv_heads: usize = config.n_kv_heads;
        const seq_len: usize = config.seq_len;
        const head_size: usize = dim / n_heads;

        var weights: Weights = undefined;

        var ptr: [*]f32 = @alignCast(@ptrCast(data));
        weights.token_embedding_table = ptr;
        ptr += vocab_size * dim;
        weights.rms_att_weight = ptr;
        ptr += n_layers * dim;
        weights.wq = ptr;
        ptr += n_layers * dim * (n_heads * head_size);
        weights.wk = ptr;
        ptr += n_layers * dim * (n_kv_heads * head_size);
        weights.wv = ptr;
        ptr += n_layers * dim * (n_kv_heads * head_size);
        weights.wo = ptr;
        ptr += n_layers * (n_heads * head_size) * dim;
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

    x: []align(simd_align) f32, // activation at current time stamp (dim,)
    xb: []align(simd_align) f32, // same, but inside a residual branch (dim,)
    xb2: []align(simd_align) f32, // an additional buffer just for convenience (dim,)
    hb: []align(simd_align) f32, // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: []align(simd_align) f32, // buffer for hidden dimension in the ffn (hidden_dim,)
    q: []align(simd_align) f32, // query (dim,)
    k: []align(simd_align) f32, // key (dim,)
    v: []align(simd_align) f32, // value (dim,)
    att: []align(simd_align) f32, // buffer for scores/attention values (n_heads, seq_len)
    logits: []align(simd_align) f32, // output logits
    logits_indexed: []align(simd_align) IndexedF32, // logits with index for top_p sampling
    // kv cache
    key_cache: []align(simd_align) f32, // (layer, seq_len, dim)
    value_cache: []align(simd_align) f32, // (layer, seq_len, dim)

    fn init(allocator: Allocator, config: *const Config) !Self {
        const kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        return Self{
            .x = try allocator.alignedAlloc(f32, simd_align, config.dim),
            .xb = try allocator.alignedAlloc(f32, simd_align, config.dim),
            .xb2 = try allocator.alignedAlloc(f32, simd_align, config.dim),
            .hb = try allocator.alignedAlloc(f32, simd_align, config.hidden_dim),
            .hb2 = try allocator.alignedAlloc(f32, simd_align, config.hidden_dim),
            .q = try allocator.alignedAlloc(f32, simd_align, config.dim),
            .k = try allocator.alignedAlloc(f32, simd_align, kv_dim),
            .v = try allocator.alignedAlloc(f32, simd_align, kv_dim),
            .att = try allocator.alignedAlloc(f32, simd_align, config.n_heads * config.seq_len),
            .logits = try allocator.alignedAlloc(f32, simd_align, config.vocab_size),
            .logits_indexed = try allocator.alignedAlloc(IndexedF32, simd_align, config.vocab_size),
            .key_cache = try allocator.alignedAlloc(f32, simd_align, config.n_layers * config.seq_len * kv_dim),
            .value_cache = try allocator.alignedAlloc(f32, simd_align, config.n_layers * config.seq_len * kv_dim),
        };
    }

    fn deinit(self: *Self, allocator: Allocator) void {
        inline for (std.meta.fields(Self)) |f| {
            allocator.free(@field(self, f.name));
        }
        self.* = undefined;
    }
};

/// Tokens, their scores, and the max token length. Supports initialization
/// from a file and encoding text into tokens via the `encode` method.
const Tokenizer = struct {
    const Self = @This();

    tokens: [][]u8,
    scores: []f32,
    max_token_len: u32,

    fn fromFile(path: []const u8, vocab_size: usize, allocator: Allocator) !Tokenizer {
        var token_file = try std.fs.cwd().openFile(path, .{});
        defer token_file.close();
        var buf_reader = std.io.bufferedReader(token_file.reader());
        const tokens = try Tokenizer.init(buf_reader.reader(), allocator, vocab_size);
        return tokens;
    }

    fn init(reader: anytype, allocator: Allocator, vocab_size: usize) !Tokenizer {
        var tokens: Tokenizer = undefined;
        tokens.tokens = try allocator.alloc([]u8, vocab_size);
        tokens.scores = try allocator.alloc(f32, vocab_size);
        tokens.max_token_len = try reader.readInt(@TypeOf(tokens.max_token_len), .little);

        for (0..vocab_size) |i| {
            tokens.scores[i] = @bitCast(try reader.readInt(u32, .little));
            const token_len = try reader.readInt(u32, .little);
            tokens.tokens[i] = try allocator.alloc(u8, token_len);
            try reader.readNoEof(tokens.tokens[i]);
        }

        return tokens;
    }

    fn deinit(self: *const Self, allocator: Allocator) void {
        for (self.tokens) |token| {
            allocator.free(token);
        }
        allocator.free(self.tokens);
        allocator.free(self.scores);
    }

    /// Given a string, find the index of the token that matches it exactly. If
    /// no token matches, returns none.
    fn lookup(self: *const Self, str: []const u8) ?u32 {
        for (self.tokens, 0..) |token, i| {
            if (std.mem.eql(u8, token, str)) {
                return @intCast(i);
            }
        }
        return null;
    }

    /// Given a string, returns the encoding as a list of tokens. You are
    /// responsible for freeing the returned list.
    fn encode(self: *const Tokenizer, input: []const u8, allocator: Allocator) ![]u32 {
        var token_buf: []u32 = try allocator.alloc(u32, input.len); // worst case is every byte is a token

        const max_allowed_token_len = 128;
        if (self.max_token_len * 2 > max_allowed_token_len) { // x2 for concat
            return error.TokensTooLong;
        }

        // need an allocator for doing string concatenation, used fixed buffer
        // allocator so we don't need to allocate any memory outside the stack
        var buffer: [max_allowed_token_len]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&buffer);
        const fixed_allocator = fba.allocator();

        var utf_encoded_buffer: [4]u8 = undefined;
        var idx: usize = 0;
        var token_end_idx: usize = 0;
        while (idx < input.len) {
            const utf_len = try std.unicode.utf8ByteSequenceLength(input[idx]);
            const codepoint: u21 = try std.unicode.utf8Decode(input[idx..][0..utf_len]);
            const encoded_len = try std.unicode.utf8Encode(codepoint, &utf_encoded_buffer);
            token_buf[token_end_idx] = self.lookup(utf_encoded_buffer[0..encoded_len]) orelse {
                return error.TokenNotFound;
            };
            token_end_idx += 1; // we have one more token now
            idx += utf_len; // skip over the utf8 sequence
        }

        while (true) {
            var best_score: f32 = -1e10;
            var best_id: u32 = 0;
            var best_idx: ?usize = null;

            // find the best token to merge
            for (0..token_end_idx - 1) |i| {
                // check if we are able to merge the token at i with the next token
                const catted = try std.mem.concat(fixed_allocator, u8, &[_][]u8{
                    self.tokens[token_buf[i]],
                    self.tokens[token_buf[i + 1]],
                });
                defer fixed_allocator.free(catted);
                if (self.lookup(catted)) |token_id| {
                    if (self.scores[token_id] > best_score) {
                        best_score = self.scores[token_id];
                        best_id = token_id;
                        best_idx = i;
                    }
                }
            }

            if (best_idx) |best| {
                // merge the best token and shift the rest of the tokens down
                token_buf[best] = best_id;
                std.mem.copyForwards(u32, token_buf[best + 1 ..], token_buf[best + 2 .. token_end_idx]);
                token_end_idx -= 1;
            } else {
                // if we didn't find any tokens to merge, we are done
                break;
            }
        }

        if (!allocator.resize(token_buf, token_end_idx)) {
            return error.OutOfMemory;
        }
        return token_buf[0..token_end_idx];
    }
};

fn transformer(token: usize, pos: usize, config: *const Config, s: *RunState, w: *const Weights) void {
    // convenience variables
    const dim: usize = config.dim;
    const hidden_dim = config.hidden_dim;
    const head_size = dim / config.n_heads;
    const kv_dim = (dim * config.n_kv_heads) / config.n_heads;
    const kv_mul = config.n_heads / config.n_kv_heads; // kv sharing in mutliquery attention
    const x = s.x;

    // copy the token embedding into x
    const embedding_row = w.token_embedding_table[token * dim ..][0..dim];
    @memcpy(x, embedding_row);

    // pluck out the "pos" row of the freq_cis real and imaginary parts
    // const freq_cis_real_row = w.freq_cis_real[pos * head_size / 2 ..][0 .. head_size / 2];
    // const freq_cis_imag_row = w.freq_cis_imag[pos * head_size / 2 ..][0 .. head_size / 2];

    // forward all the layers
    for (0..config.n_layers) |l| {
        // attention rmsnorm
        rmsnorm(s.xb, x, w.rms_att_weight[l * dim ..][0..dim]);

        // qkv
        if (kv_dim == dim) {
            matmul_fused(3, [_][]f32{ s.q, s.k, s.v }, s.xb, [_][]f32{
                w.wq[l * dim * dim ..][0 .. dim * dim],
                w.wk[l * dim * kv_dim ..][0 .. dim * kv_dim],
                w.wv[l * dim * kv_dim ..][0 .. dim * kv_dim],
            });
        } else {
            matmul(s.q, s.xb, w.wq[l * dim * dim ..][0 .. dim * dim]);
            matmul_fused(2, [_][]f32{ s.k, s.v }, s.xb, [_][]f32{
                w.wk[l * dim * kv_dim ..][0 .. dim * kv_dim],
                w.wv[l * dim * kv_dim ..][0 .. dim * kv_dim],
            });
        }

        // // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
        // for (0..2) |v| {
        //     const vec = if (v == 0) s.q else s.k;
        //     const vec_size = if (v == 0) dim else kv_dim;
        //     var i: usize = 0;
        //     while (i < vec_size) : (i += 2) {
        //         const v0 = vec[i];
        //         const v1 = vec[i + 1];
        //         const fcr = freq_cis_real_row[(i % head_size) / 2];
        //         const fci = freq_cis_imag_row[(i % head_size) / 2];
        //         vec[i] = v0 * fcr - v1 * fci;
        //         vec[i + 1] = v0 * fci + v1 * fcr;
        //     }
        // }
        var i: usize = 0;
        while (i < dim) : (i += 2) {
            const head_dim: f32 = @floatFromInt(i % head_size);
            const freq = 1.0 / std.math.pow(f32, 10000.0, head_dim / (@as(f32, @floatFromInt(head_size))));
            const val: f32 = @as(f32, @floatFromInt(pos)) * freq;
            const fcr = std.math.cos(val);
            const fci = std.math.sin(val);
            const rotn: usize = if (i < kv_dim) 2 else 1; // how many vectors? 2 = q & k, 1 = q only
            for (0..rotn) |v| {
                const vec = if (v == 0) s.q else s.k; // the vector to rotate (query or key)
                const v0 = vec[i];
                const v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // save key,value at the current timestep to our kv cache
        const loff = l * config.seq_len * kv_dim; // kv cache offset
        const key_cache_row = s.key_cache[loff + pos * kv_dim ..][0..kv_dim];
        const value_cache_row = s.value_cache[loff + pos * kv_dim ..][0..kv_dim];
        @memcpy(key_cache_row, s.k);
        @memcpy(value_cache_row, s.v);

        // attention
        for (0..config.n_heads) |h| {
            // get the query vector for this head
            const q = s.q[h * head_size ..][0..head_size];
            // attention scores
            const att = s.att[h * config.seq_len ..][0..config.seq_len];
            // iterate over the timesteps, including the current one
            for (0..pos + 1) |t| {
                // get the key for this timestep
                const k = s.key_cache[loff + t * kv_dim + (h / kv_mul) * head_size ..][0..head_size];
                // attn score as the dot of q and k
                var score: f32 = vector_dot_product(q, k);
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
                const v = s.value_cache[loff + t * kv_dim + (h / kv_mul) * head_size ..][0..head_size];
                // get the attention weight for this timestep
                const a = att[t];
                // accumulate the weighted value vector into xb
                vector_weighted_sum(xb, v, a);
            }
        }

        // final matmul to get the output of attention
        matmul(s.xb2, s.xb, w.wo[l * dim * dim ..][0 .. dim * dim]);

        // residual connection back into x
        accum(x, s.xb2);

        // ffn rmsnorm
        rmsnorm(s.xb, x, w.rms_ffn_weight[l * dim ..][0..dim]);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        // matmul(s.hb, s.xb, w.w1[l * dim * hidden_dim ..][0 .. dim * hidden_dim]);
        // matmul(s.hb2, s.xb, w.w3[l * dim * hidden_dim ..][0 .. dim * hidden_dim]);
        // fused version of the above
        matmul_fused(2, [_][]f32{ s.hb, s.hb2 }, s.xb, [_][]f32{
            w.w1[l * dim * hidden_dim ..][0 .. dim * hidden_dim],
            w.w3[l * dim * hidden_dim ..][0 .. dim * hidden_dim],
        });

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for (s.hb) |*v| {
            v.* = v.* * (1.0 / (1.0 + std.math.exp(-v.*)));
        }

        // elementwise multiply with w3(x)
        vector_mul(s.hb, s.hb2);

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
    // // This one function accounts for ~90% of the total runtime.
    // const d = xout.len;
    // const n = x.len;
    // assert(w.len == n * d);
    // assert(w.len > 0);
    //
    // // unrolling doesn't seem to help
    // for (0..d) |i| {
    //     const wrow = w[i * n ..][0..n]; // row i of W
    //     xout[i] = vector_dot_product(wrow, x);
    // }
    matmul_fused(1, [_][]f32{xout}, x, [_][]const f32{w});
}

/// Computes the vector addition of two vectors and then accumulates the result
/// into a scalar. Handles the case where the vector length is not a multiple
/// of the SIMD vector width.
fn vector_dot_product(x: []const f32, y: []const f32) f32 {
    assert(x.len == y.len);
    const vector_width = DEFAULT_VECTOR_WIDTH;
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

/// Does matrix vector multiplication using comptime to dynamically generate the fused steps.
fn matmul_fused(comptime N: usize, outs: [N][]f32, x: []const f32, ws: [N][]const f32) void {
    if (N == 0) @compileError("N must be greater than 0");
    // go through and check that all the dimensions are correct
    inline for (0..N) |i| {
        assert(outs[i].len > 0);
        assert(ws[i].len > 0);
        assert(ws[i].len == x.len * outs[i].len);
        if (i > 0) {
            assert(outs[i].len == outs[i - 1].len);
            assert(ws[i].len == ws[i - 1].len);
        }
    }

    const vector_width = DEFAULT_VECTOR_WIDTH;
    const vec_len = x.len / vector_width;
    const vec_rem = x.len % vector_width;

    const d = outs[0].len;
    const n = x.len;

    for (0..d) |i| {
        // pick out rows of W
        var wrows: [N][]const f32 = undefined;
        inline for (0..N) |j| {
            wrows[j] = ws[j][i * n ..][0..n];
        }

        // Initialize sums
        var sums: [N]@Vector(vector_width, f32) = [1]@Vector(vector_width, f32){@splat(0.0)} ** N;

        var offset: usize = 0;
        for (0..vec_len) |_| {
            const xvec: @Vector(vector_width, f32) = x[offset..][0..vector_width].*;
            inline for (0..N) |j| {
                const wvec: @Vector(vector_width, f32) = wrows[j][offset..][0..vector_width].*;
                sums[j] += xvec * wvec;
            }
            offset += vector_width;
        }

        // process remaining elements with scalar ops
        var sums_rem: [N]f32 = [1]f32{0.0} ** N;
        for (0..vec_rem) |a| {
            inline for (0..N) |j| {
                sums_rem[j] += x[offset + a] * wrows[j][offset + a];
            }
        }

        // reduce SIMD vector to scalar
        inline for (0..N) |j| {
            outs[j][i] = @reduce(.Add, sums[j]) + sums_rem[j];
        }
    }
}

/// Computes vector vector multiplication elementwise and stores the result in the first vector.
fn vector_mul(x: []f32, y: []const f32) void {
    assert(x.len == y.len);
    const vector_width = DEFAULT_VECTOR_WIDTH;
    const vec_len = x.len / vector_width; // num of SIMD vectors
    const vec_rem = x.len % vector_width; // num of f32 in the last SIMD vector

    // do the bulk of the work with SIMD
    var offset: usize = 0;
    for (0..vec_len) |_| {
        var xvec: @Vector(vector_width, f32) = x[offset..][0..vector_width].*;
        const yvec: @Vector(vector_width, f32) = y[offset..][0..vector_width].*;
        xvec *= yvec;
        x[offset..][0..vector_width].* = xvec;
        offset += vector_width;
    }

    // handle the last few elements with normal scalar ops
    for (0..vec_rem) |i| {
        x[offset + i] *= y[offset + i];
    }
}

/// Performs a weighted vector sum operation using SIMD for efficiency.
/// The operation performed is xout = xout + x * y where x is a vector and y is a scalar.
fn vector_weighted_sum(xout: []f32, x: []const f32, y: f32) void {
    assert(xout.len == x.len);
    const vector_width = DEFAULT_VECTOR_WIDTH;
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
    const random = prng.random();
    const r = random.float(f32);

    var cdf: f32 = 0.0;
    for (x, 0..) |val, i| {
        cdf += val;
        if (r < cdf) {
            return i;
        }
    }
    return x.len - 1;
}

const IndexedF32 = struct {
    index: u32,
    value: f32,

    fn desc(_: void, a: IndexedF32, b: IndexedF32) bool {
        return a.value > b.value;
    }
};

/// Top-p (nucleus) sampling. Samples from the smallest set of tokens whose
/// cumulative probability mass exceeds the probability p.
fn sample_top_p(logits: []f32, p: f32, logits_index: []IndexedF32) usize {
    assert(logits.len > 0);
    assert(p > 0.0 and p <= 1.0);
    assert(logits.len == logits_index.len);

    // elements smaller than (1 - p) / (n - 1) cannot be part of the result
    // and can be filtered out directly
    const cutoff: f32 = (1 - p) / (@as(f32, @floatFromInt(logits.len)) - 1);
    var num_to_sort: usize = 0;
    for (0..logits.len) |i| {
        assert(i < std.math.maxInt(u32));
        if (logits[i] >= cutoff) {
            logits_index[num_to_sort].value = logits[i];
            logits_index[num_to_sort].index = @intCast(i);
            num_to_sort += 1;
        }
    }
    assert(num_to_sort > 0);

    // sort the remaining elements
    std.sort.pdq(IndexedF32, logits_index[0..num_to_sort], {}, IndexedF32.desc);

    // find the cutoff index
    var cumulative_prob: f32 = 0.0;
    var cutoff_index: usize = num_to_sort - 1; // default to last element
    for (0..num_to_sort) |i| {
        cumulative_prob += logits_index[i].value;
        if (cumulative_prob > p) {
            cutoff_index = i;
            break;
        }
    }

    // sample from the cutoff index
    const random = prng.random();
    const r = random.float(f32) * cumulative_prob;
    var cdf: f32 = 0.0;
    for (0..cutoff_index + 1) |i| {
        cdf += logits_index[i].value;
        if (r < cdf) {
            return logits_index[i].index;
        }
    }
    return logits_index[cutoff_index].index;
}

const usage_text: []const u8 =
    \\Usage:   llama2 <checkpoint> [options]
    \\Example: llama2 checkpoint.bin -n 256 -i "Once upon a time"
    \\Options:
    \\ -h, --help                print this help message
    \\ -t, --temperature <float> temperature, default 1.0 (0.0, 1]
    \\ -p, --top-p <float>       p value in top-p (nucleus) sampling. default 0.9, 0 || 1 = off
    \\ -n, --seq-len <int>       number of steps to run for, default 256. 0 = max_seq_len
    \\ -i, --input <string>      input text for the prompt, default ""
    \\ -s, --seed <int>          random seed, default to time
    \\ -v, --verbose             print model info and tokens/s
    \\ -z, --tokenizer <path>    path to the tokenizer to use, default to "tokenizer.bin"
    \\
;

var prng: std.Random.DefaultPrng = undefined;
var verbose: bool = false;
fn log(comptime format: []const u8, args: anytype) void {
    if (verbose) {
        std.debug.print(format, args);
    }
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const stdout = std.io.getStdOut().writer();

    const args = try std.process.argsAlloc(allocator);
    if (args.len < 2) {
        try stdout.writeAll(usage_text);
        return;
    }

    var bin_path: ?[]const u8 = null;
    var input: ?[]const u8 = null;
    var temperature: f32 = 1.0;
    var top_p: f32 = 0.9;
    var seq_len: usize = 0;
    var tokenizer_path: []const u8 = "tokenizer.bin";
    prng = std.Random.DefaultPrng.init(@truncate(@as(u128, @bitCast(std.time.nanoTimestamp()))));

    // parse args
    var arg_i: usize = 1;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try stdout.writeAll(usage_text);
            return std.process.cleanExit();
        }
        if (!std.mem.startsWith(u8, arg, "-")) {
            if (bin_path) |_| {
                std.debug.print("error: multiple checkpoint paths specified\n", .{});
                std.process.exit(1);
            } else {
                bin_path = arg;
            }
        } else if (std.mem.eql(u8, arg, "-t") or std.mem.eql(u8, arg, "--temperature")) {
            arg_i += 1;
            if (arg_i >= args.len) {
                std.debug.print("error: missing argument for temperature\n", .{});
                std.process.exit(1);
            }
            temperature = std.fmt.parseFloat(f32, args[arg_i]) catch |err| {
                std.debug.print("unable to parse --temperature argument '{s}': {s}\n", .{
                    args[arg_i], @errorName(err),
                });
                std.process.exit(1);
            };
            // temperature = std.math.clamp(temperature, 0.0, 1.0); // TODO: clamp?
        } else if (std.mem.eql(u8, arg, "-n") or std.mem.eql(u8, arg, "--seq-len")) {
            arg_i += 1;
            if (arg_i >= args.len) {
                std.debug.print("error: missing argument for seq-len\n", .{});
                std.process.exit(1);
            }
            seq_len = std.fmt.parseInt(usize, args[arg_i], 10) catch |err| {
                std.debug.print("unable to parse --seq-len argument '{s}': {s}\n", .{
                    args[arg_i], @errorName(err),
                });
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--top-p")) {
            arg_i += 1;
            if (arg_i >= args.len) {
                std.debug.print("error: missing argument for top-p\n", .{});
                std.process.exit(1);
            }
            top_p = std.fmt.parseFloat(f32, args[arg_i]) catch |err| {
                std.debug.print("unable to parse --top-p argument '{s}': {s}\n", .{
                    args[arg_i], @errorName(err),
                });
                std.process.exit(1);
            };
            top_p = std.math.clamp(top_p, 0.0, 1.0);
        } else if (std.mem.eql(u8, arg, "-i") or std.mem.eql(u8, arg, "--input")) {
            arg_i += 1;
            if (arg_i >= args.len) {
                std.debug.print("error: missing argument for input\n", .{});
                std.process.exit(1);
            }
            input = args[arg_i];
        } else if (std.mem.eql(u8, arg, "-z") or std.mem.eql(u8, arg, "--tokenizer")) {
            arg_i += 1;
            if (arg_i >= args.len) {
                std.debug.print("error: missing argument for tokenizer\n", .{});
                std.process.exit(1);
            }
            tokenizer_path = args[arg_i];
        } else if (std.mem.eql(u8, arg, "-s") or std.mem.eql(u8, arg, "--seed")) {
            arg_i += 1;
            if (arg_i >= args.len) {
                std.debug.print("error: missing argument for seed\n", .{});
                std.process.exit(1);
            }
            const seed = std.fmt.parseInt(u64, args[arg_i], 10) catch |err| {
                std.debug.print("unable to parse --seed argument '{s}': {s}\n", .{
                    args[arg_i], @errorName(err),
                });
                std.process.exit(1);
            };
            prng = std.Random.DefaultPrng.init(seed);
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
            verbose = true;
        } else {
            std.debug.print("error: unknown argument '{s}'\n", .{arg});
            try stdout.writeAll(usage_text);
            return std.process.cleanExit();
        }
    }

    // read the config from the checkpoint
    var checkpoint = try std.fs.cwd().openFile(bin_path.?, .{});
    // close by hand
    var config_read: ConfigReader = try checkpoint.reader().readStruct(ConfigReader);
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    const shared_weights: bool = config_read.vocab_size > 0;
    config_read.vocab_size = @intCast(@abs(config_read.vocab_size));
    const file_size = (try checkpoint.stat()).size;
    const config = config_read.config(); // convert to usize version

    log("config: {any}\n", .{config});
    log("shared weights: {any}\n", .{shared_weights});
    log("temperature: {d}\n", .{temperature});
    log("top-p: {d}\n", .{top_p});
    log("SIMD vector size: {d}\n", .{DEFAULT_VECTOR_WIDTH});
    log("\n", .{});

    const data: []align(mem.page_size) u8 = blk: {
        const weights_size: usize = file_size - @sizeOf(ConfigReader);
        const buffer = try allocator.alignedAlloc(u8, mem.page_size, weights_size);
        const read_len = try checkpoint.readAll(buffer);
        if (read_len != weights_size) {
            std.debug.print("error: failed to read checkpoint file\n", .{});
            std.process.exit(1);
        }
        checkpoint.close();
        break :blk buffer;
        // mmap seems slower
        // break :blk try std.os.mmap(null, file_size, std.os.PROT.READ, std.os.MAP.PRIVATE, mapped_checkpoint.handle, 0);
    };
    defer allocator.free(data);

    const weights = Weights.init(&config, data, shared_weights);

    // load the tokens for the model
    const tokenizer = try Tokenizer.fromFile(tokenizer_path, config.vocab_size, allocator);
    defer tokenizer.deinit(allocator);

    // initialize the run state for inference
    var state = try RunState.init(allocator, &config);
    defer state.deinit(allocator);

    // encode the prompt
    var prompt: ?[]u32 = null;
    var prompt_len: usize = 0; // avoid the double if later
    defer if (prompt) |p| allocator.free(p);
    if (input) |in| {
        const encoded_input = try tokenizer.encode(in, allocator);
        prompt_len = encoded_input.len;
        prompt = encoded_input;
    }

    var next: usize = undefined; // the next token as predicted by the model
    var token: usize = 1; // 1 = <BOS> for llama2
    var timer: ?std.time.Timer = null;

    // adjust the sequence length if needed
    seq_len = if (seq_len == 0) config.seq_len else seq_len;
    seq_len = std.math.clamp(seq_len, 1, config.seq_len); // clamp to seq_len
    var pos: usize = 0; // the current position in the sequence
    while (pos < seq_len) : (pos += 1) {
        transformer(token, pos, &config, &state, &weights);

        // if we have a prompt, we need to feed it in
        if (pos < prompt_len) {
            next = prompt.?[pos];
        } else {
            if (temperature == 0.0) {
                next = argmax(state.logits);
            } else {
                if (temperature != 1.0) {
                    for (state.logits) |*val| val.* /= temperature;
                }
                softmax(state.logits);
                next = if (top_p == 0.0 or top_p == 1.0)
                    sample(state.logits)
                else
                    sample_top_p(state.logits, top_p, state.logits_indexed);
            }
        }

        // 1 = <BOS> which ends the sequence
        if (next == 1) {
            break;
        }

        // print the token, at the start of the sequence we don't want to print the space
        const token_str = if (token == 1 and tokenizer.tokens[next][0] == ' ')
            tokenizer.tokens[next][1..]
        else
            tokenizer.tokens[next];

        // handle case when tokens are raw bytes
        if (isRawByte(token_str)) |byte| {
            try stdout.print("{c}", .{byte});
            token = next;
            continue;
        } else {
            try stdout.print("{s}", .{token_str});
        }

        token = next;

        // if timer is null, start it
        if (timer == null) {
            timer = try std.time.Timer.start();
        }
    }
    const time = timer.?.read();
    const tokens_per_ms = @as(f64, @floatFromInt(pos - 1)) / @as(f64, @floatFromInt(time / std.time.ns_per_ms));
    const tokens_per_sec: u32 = @intFromFloat(tokens_per_ms * 1000.0);

    // print tokens per second
    log("\n\n{d} tokens per second\n", .{tokens_per_sec});
}

/// Matches the pattern <0xXX> where XX is a hex number and
/// returns the byte value of the hex number.
fn isRawByte(input: []const u8) ?u8 {
    if (input.len != 6) return null;
    if (input[0] != '<' or input[1] != '0' or input[2] != 'x' or input[5] != '>') return null;
    var byte: u8 = 0;
    for (input[3..5]) |c| {
        byte *= 16;
        if (c >= '0' and c <= '9') {
            byte += c - '0';
        } else if (c >= 'a' and c <= 'f') {
            byte += c - 'a' + 10;
        } else if (c >= 'A' and c <= 'F') {
            byte += c - 'A' + 10;
        } else {
            return null;
        }
    }
    if (std.ascii.isPrint(byte) or std.ascii.isWhitespace(byte)) {
        return byte;
    } else {
        return null;
    }
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
    const y: f32 = 3.0;
    var xout = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };

    vector_weighted_sum(&xout, &x, y);
    for (0..xout.len) |i| {
        const expected = (x[i] * y) + x[i];
        try std.testing.expect((xout[i] - expected) < 0.0001);
    }
}

test "softmax" {
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    softmax(&x);
    var sum: f32 = 0.0;
    for (0..x.len) |i| {
        sum += x[i];
    }
    try std.testing.expect(sum == 1.0);
}

test "bpe" {
    var allocator = std.testing.allocator;
    const tokenizer = try Tokenizer.fromFile("tokenizer.bin", 32000, allocator);
    defer tokenizer.deinit(allocator);

    try std.testing.expect(tokenizer.lookup("æ") == 233);
    try std.testing.expect(std.mem.eql(u8, tokenizer.tokens[100], "a"));
    try std.testing.expect(tokenizer.max_token_len == 27);
    try std.testing.expect(tokenizer.tokens.len == tokenizer.scores.len);
    try std.testing.expect(tokenizer.tokens.len == 32000);
    try std.testing.expect(tokenizer.lookup("a") == 100);

    const input: []const u8 = "A man dying of thirst is suddenly a mineral water critic?";
    const expected_tokenization: []const u32 = &[_]u32{ 68, 767, 27116, 310, 266, 765, 338, 11584, 263, 1375, 13537, 4094, 11164, 66 };
    const tokenization = try tokenizer.encode(input, allocator);
    defer allocator.free(tokenization);
    try std.testing.expect(tokenization.len == expected_tokenization.len);
    for (tokenization, 0..) |token, i| {
        try std.testing.expect(token == expected_tokenization[i]);
    }
    const utf_input: []const u8 = "中";
    const utf_expected_tokens: []const u32 = &[_]u32{30275};
    const utf_tokenization = try tokenizer.encode(utf_input, allocator);
    defer allocator.free(utf_tokenization);
    try std.testing.expect(utf_tokenization.len == utf_expected_tokens.len);
    for (utf_tokenization, 0..) |token, i| {
        try std.testing.expect(token == utf_expected_tokens[i]);
    }
}
