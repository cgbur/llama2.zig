# llama2.zig

This is the Zig version of [llama2.c](https://github.com/karpathy/llama2.c) by
Andrej Karpathy. This runs inference for the
[llama2](https://github.com/facebookresearch/llama) model architecture recently
published by Meta.

This is a work in progress side project and as such will likely not be feature
complete or always up to date with the latest llama2.c. Pull requests are
welcome. The long term goal is to have a fast, portable, and easy to use
implementation of the llama2 model architecture. An emphasis is placed on
simplicity and readability of the code where it does not impact performance
significantly. A SIMD implementation of the `matmul` function has been
implemented using the Zig `@Vector` feature for a ~4x speedup. See the
[performance](#performance) section for more details.

`stories15.bin` is a model checkpoint file for a 15M parameter model trained on
the tiny stories dataset. See the llama2.c repo for how this is generated. The
tokenizer.bin file from that repo is out of date and will be ported here
eventually.

## Usage

After cloning the repo, you can run inference with:

```sh
zig build run -Doptimize=ReleaseFast -- stories15M.bin 0.9
```

# Performance

The following benchmarks were run on a AMD Ryzen 9 5900X 12-Core Processor.

| Implementation                                    | Tokens/s |
| ------------------------------------------------- | -------- |
| llama2.c `make run`                               | 116      |
| llama2.c `make runfast`                           | 375      |
| llama2.zig `zig build run -Doptimize=ReleaseFast` | 447      |

## Todo

- \[ \] Support tokenization for prompting support
- \[ \] Add the python to generate the checkpoints here
- \[ \] Parallelize multi-head attention

## Contributing

Contributions are welcome. Please open an issue or pull request.
