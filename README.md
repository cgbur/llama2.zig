# llama2.zig

This is the Zig version of [llama2.c](https://github.com/karpathy/llama2.c) by
Andrej Karpathy. This runs inference for the
[llama2](https://github.com/facebookresearch/llama) model architecture recently
published by Meta.

This is a work in progress side project that is incomplete and likely never to
be feature complete or on par with the original C version. Already it has fallen
behind in the few days since I started it.

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

## Contributing

Contributions are welcome. Please open an issue or pull request.
