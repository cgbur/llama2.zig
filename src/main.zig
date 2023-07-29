const std = @import("std");
const Allocator = std.mem.Allocator;

/// Configuration for the model
const Config = extern struct {
    dim: i32, // transformer dimension
    hidden_dim: i32, // for ffn layers
    n_layers: i32, // number of layers
    n_heads: i32, // number of query heads
    n_kv_heads: i32, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    seq_len: i32, // max sequence length
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

    fn init(config: *Config, data: []u8, shared_weights: bool) Weights {
        const vocab_size: usize = @intCast(config.vocab_size);
        const dim: usize = @intCast(config.dim);
        const hidden_dim: usize = @intCast(config.hidden_dim);
        const n_layers: usize = @intCast(config.n_layers);
        const n_heads: usize = @intCast(config.n_heads);
        const seq_len: usize = @intCast(config.seq_len);

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
        //
        // for (0..vocab_size) |i| {
        //     std.debug.print("{any}\n", .{tokens.tokens[i]});
        //     std.debug.print("{s}\n", .{tokens.tokens[i]});
        // }

        return tokens;
    }
};

/// An array of tokens which are strings of bytes
const bin_path = "stories15M.bin";

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var checkpoint = try std.fs.cwd().openFile(bin_path, .{}); // close this by hand
    var config: Config = try checkpoint.reader().readStruct(Config);
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    const shared_weights: bool = config.vocab_size > 0;
    config.vocab_size = try std.math.absInt(config.vocab_size);
    const file_size = (try checkpoint.stat()).size;
    checkpoint.close();

    std.debug.print("config: {any}\n", .{config});
    std.debug.print("shared weights: {any}\n", .{shared_weights});

    const mapped_checkpoint = try (std.fs.cwd().openFile(bin_path, .{}));
    defer mapped_checkpoint.close();
    const data: []u8 = try std.os.mmap(null, file_size, std.os.linux.PROT.READ, std.os.linux.MAP.PRIVATE, mapped_checkpoint.handle, 0);
    const weights = Weights.init(&config, data[@sizeOf(Config)..], shared_weights);
    _ = weights;

    const steps = config.seq_len;
    _ = steps;

    const tokens = tokenblk: {
        var token_file = try std.fs.cwd().openFile("tokenizer.bin", .{});
        defer token_file.close();
        var buf_reader = std.io.bufferedReader(token_file.reader());
        const tokens = try Tokens.init(buf_reader.reader(), allocator, @intCast(config.vocab_size));
        break :tokenblk tokens;
    };
    _ = tokens;

    // var state = try RunState.init(allocator, &config);
    // _ = state;
}
