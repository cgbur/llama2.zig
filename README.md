# llama2.zig CUDA

<p align="center">
  <img src="assets/llama_and_ziggy.jpg" width="300" height="300" alt="Cute Llama">
</p>

This branch runs [llama2.c](https://github.com/karpathy/llama2.c) checkpoints on
NVIDIA GPUs. The portable CPU implementation lives on the
[`main`](https://github.com/cgbur/llama2.zig/tree/main) branch.

The implementation deliberately has one inference path: FP16 checkpoint
weights, FP32 activations and accumulation, fused CUDA kernels, and CUDA Graph
replay. Zig handles checkpoint loading, tokenization, sampling, and output; the
complete transformer forward pass and greedy argmax stay on the GPU.

It supports prompts, temperature, top-p sampling, custom tokenizers,
multi-query attention, and configurable sequence lengths.

## Requirements

- An NVIDIA GPU and driver
- Zig 0.16.0
- CUDA 12.9 or newer for the default RTX 5090 target
- Nix with flakes enabled, or equivalent `nvcc`, `cudart`, and `pkg-config`

The included Nix shell supplies the CUDA dependencies. CUDA packages use the
NVIDIA EULA; `flake.nix` enables unfree packages for this shell.

## Build and run

```sh
nix develop
zig build -Doptimize=ReleaseFast
zig-out/bin/llama2 stories15M.bin
```

The build targets `sm_120` for the RTX 5090 by default. Override it for another
GPU:

```sh
zig build -Doptimize=ReleaseFast -Dcuda-arch=sm_89
```

A prompt and greedy sampling can be selected as usual:

```sh
zig-out/bin/llama2 stories15M.bin -i "Once upon a time" -t 0 -v
```

Run `zig-out/bin/llama2 --help` for all options.

## Execution model

At startup, Zig reads the FP32 checkpoint, uploads it once, and converts the
weights to FP16. Activations, logits, attention storage, and the KV cache remain
on the GPU. One token's fused kernel sequence is captured as a CUDA Graph and
replayed for every autoregressive step. Greedy generation copies only the next
32-bit token back to Zig. Temperature and top-p sampling copy the logits back
and use the original Zig sampler.

On an RTX 5090, the included 15M-parameter checkpoint measured about 4,000
greedy tokens/s over 256-token runs. Model loading, weight conversion, and the
first warm-up token are outside the existing tokens/s timer.

FP16 weights are intentionally lossy. They preserved the included checkpoint's
greedy output in testing but can change sampled output relative to the FP32 CPU
implementation.

## Credits

- [Andrej Karpathy](https://github.com/karpathy) for `llama2.c`
- [Meta](https://github.com/facebookresearch/llama) for Llama 2
