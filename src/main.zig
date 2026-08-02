const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const assert = std.debug.assert;

const CudaBackend = opaque {};
const CudaConfig = extern struct {
    dim: i32,
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
};

extern fn llama2_cuda_create(
    config: *const CudaConfig,
    weights: [*]const f32,
    weights_count: usize,
    shared_weights: i32,
) ?*CudaBackend;
extern fn llama2_cuda_forward(
    context: *CudaBackend,
    token: i32,
    pos: i32,
    host_logits: ?[*]f32,
    host_next: *i32,
) i32;
extern fn llama2_cuda_destroy(context: *CudaBackend) void;
extern fn llama2_cuda_last_error() [*:0]const u8;

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

const SamplingState = struct {
    logits: []f32,
    logits_indexed: []IndexedF32,

    fn init(allocator: Allocator, vocab_size: usize) !SamplingState {
        const logits = try allocator.alloc(f32, vocab_size);
        errdefer allocator.free(logits);
        return .{
            .logits = logits,
            .logits_indexed = try allocator.alloc(IndexedF32, vocab_size),
        };
    }

    fn deinit(self: *SamplingState, allocator: Allocator) void {
        allocator.free(self.logits);
        allocator.free(self.logits_indexed);
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

    fn fromFile(path: []const u8, vocab_size: usize, allocator: Allocator, io: std.Io) !Tokenizer {
        const token_file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer token_file.close(io);
        var read_buffer: [4096]u8 = undefined;
        var file_reader = token_file.reader(io, &read_buffer);
        const tokens = try Tokenizer.init(&file_reader.interface, allocator, vocab_size);
        return tokens;
    }

    fn init(reader: *std.Io.Reader, allocator: Allocator, vocab_size: usize) !Tokenizer {
        var tokens: Tokenizer = undefined;
        tokens.tokens = try allocator.alloc([]u8, vocab_size);
        tokens.scores = try allocator.alloc(f32, vocab_size);
        tokens.max_token_len = try reader.takeInt(@TypeOf(tokens.max_token_len), .little);

        for (0..vocab_size) |i| {
            tokens.scores[i] = @bitCast(try reader.takeInt(u32, .little));
            const token_len = try reader.takeInt(u32, .little);
            tokens.tokens[i] = try allocator.alloc(u8, token_len);
            try reader.readSliceAll(tokens.tokens[i]);
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

        token_buf = try allocator.realloc(token_buf, token_end_idx);
        return token_buf;
    }
};

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

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_file_writer.interface;
    defer stdout.flush() catch {};

    const args = try init.minimal.args.toSlice(allocator);
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
    const now_ns = std.Io.Clock.real.now(io).nanoseconds;
    prng = std.Random.DefaultPrng.init(@truncate(@as(u96, @bitCast(now_ns))));

    // parse args
    var arg_i: usize = 1;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try stdout.writeAll(usage_text);
            return;
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
            return;
        }
    }

    // read the config from the checkpoint
    const checkpoint = try std.Io.Dir.cwd().openFile(io, bin_path.?, .{});
    // close by hand
    var checkpoint_read_buffer: [4096]u8 = undefined;
    var checkpoint_reader = checkpoint.reader(io, &checkpoint_read_buffer);
    var config_read = try checkpoint_reader.interface.takeStruct(ConfigReader, .little);
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    const shared_weights: bool = config_read.vocab_size > 0;
    config_read.vocab_size = @intCast(@abs(config_read.vocab_size));
    const file_size = (try checkpoint.stat(io)).size;
    const config = config_read.config(); // convert to usize version

    log("config: {any}\n", .{config});
    log("shared weights: {any}\n", .{shared_weights});
    log("temperature: {d}\n", .{temperature});
    log("top-p: {d}\n", .{top_p});
    log("\n", .{});

    const data: []align(std.heap.page_size_min) u8 = blk: {
        const weights_size = std.math.cast(usize, file_size - @sizeOf(ConfigReader)) orelse
            return error.FileTooBig;
        const buffer = try allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), weights_size);
        try checkpoint_reader.interface.readSliceAll(buffer);
        checkpoint.close(io);
        break :blk buffer;
        // mmap seems slower
        // break :blk try std.os.mmap(null, file_size, std.os.PROT.READ, std.os.MAP.PRIVATE, mapped_checkpoint.handle, 0);
    };
    defer allocator.free(data);

    const cuda_config = CudaConfig{
        .dim = @intCast(config.dim),
        .hidden_dim = @intCast(config.hidden_dim),
        .n_layers = @intCast(config.n_layers),
        .n_heads = @intCast(config.n_heads),
        .n_kv_heads = @intCast(config.n_kv_heads),
        .vocab_size = @intCast(config.vocab_size),
        .seq_len = @intCast(config.seq_len),
    };
    const cuda_context = llama2_cuda_create(
        &cuda_config,
        @ptrCast(@alignCast(data.ptr)),
        data.len / @sizeOf(f32),
        @intFromBool(shared_weights),
    ) orelse {
        std.debug.print("error: unable to initialize CUDA: {s}\n", .{llama2_cuda_last_error()});
        std.process.exit(1);
    };
    defer llama2_cuda_destroy(cuda_context);

    // load the tokens for the model
    const tokenizer = try Tokenizer.fromFile(tokenizer_path, config.vocab_size, allocator, io);
    defer tokenizer.deinit(allocator);

    // initialize the run state for inference
    var state = try SamplingState.init(allocator, config.vocab_size);
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
    var timer: ?std.Io.Timestamp = null;

    // adjust the sequence length if needed
    seq_len = if (seq_len == 0) config.seq_len else seq_len;
    seq_len = std.math.clamp(seq_len, 1, config.seq_len); // clamp to seq_len
    var pos: usize = 0; // the current position in the sequence
    while (pos < seq_len) : (pos += 1) {
        var gpu_next: i32 = undefined;
        const need_logits = pos >= prompt_len and temperature != 0.0;
        const result = llama2_cuda_forward(
            cuda_context,
            @intCast(token),
            @intCast(pos),
            if (need_logits) state.logits.ptr else null,
            &gpu_next,
        );
        if (result != 0) {
            std.debug.print("error: CUDA inference failed: {s}\n", .{llama2_cuda_last_error()});
            std.process.exit(1);
        }

        // if we have a prompt, we need to feed it in
        if (pos < prompt_len) {
            next = prompt.?[pos];
        } else {
            if (temperature == 0.0) {
                next = @intCast(gpu_next);
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
            timer = std.Io.Clock.awake.now(io);
        }
    }
    const elapsed_ns = timer.?.untilNow(io, .awake).nanoseconds;
    const tokens_per_sec: u32 = @intFromFloat(
        @as(f64, @floatFromInt(pos - 1)) * std.time.ns_per_s /
            @as(f64, @floatFromInt(elapsed_ns)),
    );

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
    const tokenizer = try Tokenizer.fromFile("tokenizer.bin", 32000, allocator, std.testing.io);
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
