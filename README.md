# llama2.zig

This is the Zig version of [llama2.c](https://github.com/karpathy/llama2.c) by
Andrej Karpathy. This runs inference for the
[llama2](https://github.com/facebookresearch/llama) model architecture recently
published by Meta.

This is a work in progress side project that and as such will not be feature
complete or always up to date with the latest llama2.c. Pull requests are
welcome. The long term goal is to have a fast, portable, and easy to use
implementation of the llama2 model architecture.

`stories15.bin` is a model checkpoint file for a 15M parameter model trained on
the tiny stories dataset. See the llama2.c repo for how this is generated. The
tokenizer.bin file from that repo is out of date and will be ported here
eventually.

## Usage

After cloning the repo, you can run inference with:

```sh
zig build run -Doptimize=ReleaseFast -- stories15M.bin 0.9
```

## Todo

- \[ \] Support tokenization for prompting support
- \[ \] Make parallel and faster
- \[ \] Add the python to generate the checkpoints here
- \[ \] understand why `@setFloatMode(std.builtin.FloatMode.Optimized);` is not working

## Contributing

Contributions are welcome. Please open an issue or pull request.
