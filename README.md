# llama2.zig

<p align="center">
  <img src="assets/llama_and_ziggy.jpg" width="300" height="300" alt="Cute Llama">
</p>

This is the Zig version of [llama2.c](https://github.com/karpathy/llama2.c) by
Andrej Karpathy. It runs inference for the
[llama2](https://github.com/facebookresearch/llama) model architecture recently
published by Meta.

As a work in progress side project, it may not always be feature complete or
up-to-date with the latest version of llama2.c. However, contributions and pull
requests are greatly appreciated. The ultimate goal is to create a fast,
portable, and user-friendly implementation of the llama2 model architecture.
The code prioritizes simplicity and readability without sacrificing
performance. Certain core functions have SIMD implementations using the Zig
`@Vector` feature, which provides a ~5x speed increase. For more details,
please refer to the [performance](#performance) section.

The `stories15.bin` file is a model checkpoint for a 15M parameter model that
was trained on the tiny stories dataset. The method for generating this file
can be found in the llama2.c repo. The tokenizer.bin file from that repo is
currently out of date but will be ported here in due course.

## Usage

After cloning the repo, run the following command for inference:

```sh
zig build run -Doptimize=ReleaseFast -- stories15M.bin 0.9
```

## Performance

The benchmarks provided below were executed on an AMD Ryzen 9 5900X 12-Core
Processor.

If possible all benchmarks were run with `stories15M.bin` and a temperature of
`0.9`.

## Single-threaded

| Implementation                                          | Tokens/s |
| ------------------------------------------------------- | -------- |
| llama2.zig (this repo)                                  | 525      |
| llama2.c `make runfast -march=native`                   | 511      |
| [llama2.zig](https://github.com/clebert/llama2.zig) [1] | 473      |
| llama2.c `make runfast`                                 | 375      |
| llama2.c `make run -march=native`                       | 122      |
| llama2.c `make run`                                     | 116      |
| [llama2.rs](https://github.com/gaxler/llama2.rs)        | 115      |

[1] Only argmax token selection is supported in this implementation. In
benchmarking this has shown to be faster than the sampling method.

## Multi-threaded

This implementation currently does not support multithreading so is not
included in the table below.

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
zig build run -Doptimize=ReleaseFast -- stories15M.bin 0.9
```

```sh
zig version -> 0.11.0-dev.4315+f5239677e
```

#### llama2.c

```sh
 ./run stories15M.bin 0.9
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

- \[ \] Add tokenization support for prompt handling
- \[ \] Incorporate the Python script used to generate checkpoints
- \[ \] Parallelize multi-head attention process

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
