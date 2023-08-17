# llama2.zig

<p align="center">
  <img src="assets/llama_and_ziggy.jpg" width="300" height="300" alt="Cute Llama">
</p>

This is the Zig version of [llama2.c](https://github.com/karpathy/llama2.c) by
Andrej Karpathy. It runs inference for the
[llama2](https://github.com/facebookresearch/llama) model architecture recently
published by Meta.

It currently supports:

- Inference of llama2 model checkpoints
- Temperature control
- Top-p (nucleus) sampling
- Prompt handling (bpe tokenization)
- Sequence length control
- Custom tokenizers
- Multiquery support
- Running really fast

The ultimate goal is to create a fast, portable, and user-friendly
implementation of the llama2 model architecture. The code prioritizes simplicity
and readability without sacrificing performance. Certain core functions have
SIMD implementations using the Zig `@Vector` feature, which provides a ~5x speed
increase. For more details and comparisons to other implementation, please refer
to the [performance](#performance) section.

The `stories15.bin` file is a model checkpoint for a 15M parameter model that
was trained on the tiny stories dataset. The method for generating this file
can be found in the llama2.c repo.

## Usage

After cloning the repo, run the following command for inference:

```sh
zig build -Doptimize=ReleaseFast
zig-out/bin/llama2 stories15M.bin
```

A prompt can be provided as an argument to the program:

```sh
llama2 stories15M.bin -i "Once upon a time"
```

For all of the options, run:

```sh
$ llama2 --help
Usage:   llama2 <checkpoint> [options]
Example: llama2 checkpoint.bin -n 256 -i "Once upon a time"
Options:
 -h, --help                print this help message
 -t, --temperature <float> temperature, default 1.0 (0.0, 1]
 -p, --top-p <float>       p value in top-p (nucleus) sampling. default 1.0, 0 || 1 = off
 -n, --seq-len <int>       number of steps to run for, default 256. 0 = max_seq_len
 -i, --input <string>      input text for the prompt, default ""
 -v, --verbose             print model info and tokens/s
```

## Performance

The benchmarks provided below were executed on an AMD Ryzen 9 5900X 12-Core
Processor. All speeds measurements are taken using the `stories15M.bin`
checkpoint file.

If you have an implementation you want to add to the table, please open an issue
and I'll be happy to add it. Please only submit implementations that are single
language implementations (no OpenBlas, etc.).

## Single-threaded

### Argmax sampling

- Temperature 0.0
- 256 tokens

| Implementation                                      | Tokens/s |
| --------------------------------------------------- | -------- |
| llama2.zig (this repo)                              | 612      |
| llama2.c `make runfast -march=native`               | 548      |
| [llama2.zig](https://github.com/clebert/llama2.zig) | 496      |
| llama2.c `make run -march=native`                   | 122      |
| [llama2.rs](https://github.com/gaxler/llama2.rs)    | 115      |

### Top-P sampling

- Temperature 1.0
- Top-P 0.9
- 256 tokens

| Implementation                        | Tokens/s |
| ------------------------------------- | -------- |
| llama2.zig (this repo)                | 579      |
| llama2.c `make runfast -march=native` | 241      |

## Multi-threaded

This implementation currently does not support multithreading so is not
included in the table below. This is with temperate 0.9 and no top-p.

| Implementation                                   | Tokens/s |
| ------------------------------------------------ | -------- |
| llama2.c `make runomp`                           | 1564     |
| [llama2.rs](https://github.com/gaxler/llama2.rs) | 441      |

#### llama2.zig (this repo)

The single largest speed increase came from writing a SIMD version of matrix
multiplication using the Zig `@Vector` feature. This was an immediate jump from
around 115 tokens/s to 430 tokens/s. Notable speed increases also came from:

- `comptime` magic to generate fused matrix multiplication
- Vector aligned memory allocation
- Using SIMD versions of other core functions

```sh
llama2 stories15M.bin -t 0
llama2 stories15M.bin -t 1.0 -p 0.9
llama2 stories15M.bin -t 1.0 -p 0.9 -i "Once upon a time"
```

```sh
zig version -> 0.11.0-dev.4315+f5239677e
```

#### llama2.c

```sh
 ./run stories15M.bin -t 1.0 -p 0.9
 ./run stories15M.bin -t 0.0
```

```sh
CC = gcc

.PHONY: runfast
runfast: run.c
	$(CC) -Ofast -o run run.c -lm -march=native


.PHONY: run
run: run.c
	$(CC) -O3 -o run run.c -lm -march=native


.PHONY: runomp
runomp: run.c
	$(CC) -Ofast -fopenmp -march=native run.c  -lm  -o run -march=native
```

#### llama2.rs

```sh
 RUSTFLAGS="-C target-cpu=native" cargo run -r -- stories15M.bin 0.9
 RUSTFLAGS="-C target-cpu=native" cargo run -r -F parallel -- stories15M.bin 0.9
```

```toml
[profile.release]
codegen-units = 1
lto = true
panic = "abort"
strip = "symbols"
```

## Todo

- \[ \] Parallelize multi-head attention process
- \[ \] Add support for multi-threading (this is not going well)
- \[ \] Try top-p sampling by doing multiple linear scans to avoid sorting
- \[ \] binary search the token encoder, probably not necessary
- \[ \] Add benchmarks for smaller model and tokenizer

## Contributing

Any form of contribution is welcome. Feel free to open an issue or create a
pull request. If you are contributing optimizations, please provide benchmarks
and/or performance comparisons as well as the code to reproduce them.

# Credits

- [Andrej Karpathy](https://github.com/karpathy) for the original llama2.c
  implementation
- [Foundation42](https://github.com/Foundation42) for opening a
  [PR](https://github.com/karpathy/llama2.c/pull/94/files) on the llama2.c repo
  that was the inspiration for aligned memory allocation and fused matrix
  multiplication.
- [jrudolph](https://github.com/jrudolph) for top-p sampling optimization
  [PR](https://github.com/karpathy/llama2.c/pull/276)
