# llama2.zig

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
`@Vector` feature, which provides a ~4x speed increase. For more details,
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

| Implementation                                    | Tokens/s |
| ------------------------------------------------- | -------- |
| llama2.c `make run`                               | 116      |
| llama2.c `make runfast`                           | 375      |
| llama2.zig `zig build run -Doptimize=ReleaseFast` | 466      |

## Todo

- \[ \] Add tokenization support for prompt handling
- \[ \] Incorporate the Python script used to generate checkpoints
- \[ \] Parallelize multi-head attention process

## Contributing

Any form of contribution is welcome. Feel free to open an issue or create a
pull request.
