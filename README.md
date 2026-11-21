# Compute Shaders

**Compute Shaders** is a reusable collection of custom GPU kernels, starting with Apple Silicon and PyTorch MPS.

The goal is simple: make it easy to write, test, benchmark, and distribute compute kernels that operate directly on GPU tensors.

> Early development: the library contains reusable MPS kernels for ML-phys pair features and FastSpeech2 length regulation. APIs may change before the first stable release.

## Why this project exists

Framework-provided GPU operations are convenient, but a workload can sometimes benefit from a fused kernel designed around its exact memory layout and access pattern.

Compute Shaders keeps those experiments reusable:
- Metal shader source lives in `python/computeshaders/kernels/`.
- Python wrappers live in `python/computeshaders/`.
- Correctness tests live in `tests/`.
- Benchmarks live in `benchmarks/`.
- Examples live in `examples/`.

The first backend uses `torch.mps.compile_shader()`. Kernels receive and return PyTorch MPS tensors directly; the library does not copy tensors through CPU memory.

## Requirements

- Apple Silicon Mac
- macOS supported by your PyTorch build
- Python 3.10+
- PyTorch 2.7+ with MPS support

## Install for development

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[mps,dev]'
```

## Install from GitHub

The public source can be installed directly:

```bash
python -m pip install "computeshaders[mps] @ git+https://github.com/flushwhy/compute_shaders.git"
```

This installs the optional PyTorch dependency as well. If PyTorch is already installed, use the repository URL without the `[mps]` extra.

If PyTorch is already installed in your environment, install the project without the optional `mps` extra:

```bash
python -m pip install -e .
```

## Verify the Metal kernel

```bash
PYTHONPATH=python python scripts/verify_mps.py
```

Or use the build/verification helper:

```bash
PYTHON=python3 ./metal/build.sh
```

The shader is compiled by PyTorch at runtime. This avoids requiring a separately built native extension or a precompiled library for the initial MPS backend.

GitHub's standard macOS runners may not expose a Metal GPU even when the host architecture is Apple Silicon. In that case, the hosted workflow runs the API/packaging tests and explicitly skips the hardware shader check. The full GPU tests run on a physical MPS-capable Mac.

## Example

```python
import torch
from computeshaders import MPSKernelLibrary

states = torch.randn((4, 20, 12), device="mps", dtype=torch.float32)
kernels = MPSKernelLibrary()
pair_features = kernels.build_pair_features(states)

print(pair_features.shape)  # (4, 20, 20, 28)
```

## First kernel: pair-feature assembly

Input:
- `states`: contiguous float32 tensor shaped `(batch, bodies, 12)`

Output:
- `pair_features`: float32 tensor shaped `(batch, bodies, bodies, 28)`

Each pair contains:
1. Target body state (12 values)
2. Neighbor body state (12 values)
3. Relative x/y position (2 values)
4. Relative x/y velocity (2 values)

This matches the feature layout currently assembled in ML-phys. The kernel fuses the expanded views, relative feature calculation, and concatenation into one Metal dispatch.

The feature-only kernel is intentionally forward-only. Its state tensor must not require gradients; model parameter gradients still flow normally through the PyTorch pair encoder that consumes the returned tensor.

## Fused pair projection (experimental)

The library also exposes `MPSKernelLibrary.project_pair_features(states, weight, bias)`. It constructs the same 28 pair features and applies the first linear projection in one Metal dispatch, without writing the full pair-feature tensor during the forward pass.

A custom autograd backward recomputes pair features and returns gradients for the input states, projection weights, and bias. This supports multi-step training rollouts, where later states depend on earlier predictions. It is currently experimental: on the M1 Max benchmark, the fused path was slightly slower than the PyTorch reference at the tested batch sizes. See [docs/performance.md](docs/performance.md).

## FastSpeech2 length regulation

The library includes an MPS implementation of the FastSpeech2 length regulator. It replaces the Python batch/phoneme loops and repeated `.item()` calls with a Metal dispatch. Each output frame independently finds its source phoneme using a binary search over exclusive duration prefix sums.

The forward and backward passes are custom Metal kernels. The backward pass sums each repeated frame's gradient back into its source phoneme, so the operation remains usable during model training.

```python
import torch
from computeshaders import MPSLengthRegulator

length_regulator = MPSLengthRegulator()
hidden_states = torch.randn(8, 100, 256, device="mps", dtype=torch.float32)
durations = torch.randint(1, 6, (8, 100), device="mps", dtype=torch.int32)

expanded, mel_lengths = length_regulator(hidden_states, durations)
```

Inputs must be MPS tensors: float32 states shaped `(batch, source_length, hidden_size)`, and int32/int64 durations shaped `(batch, source_length)`. Negative durations are treated as zero. Pass `max_len` when the padded output width is already known; omitting it requires one scalar synchronization to determine the output shape.

Run the correctness tests on an MPS-capable Mac:

```bash
PYTHONPATH=python python -m unittest discover -s tests -v
```

Run the loop-vs-Metal benchmark:

```bash
PYTHONPATH=python python benchmarks/benchmark_length_regulator.py --warmup 5 --repetitions 30
```

The benchmark compares against the original loop-based behavior, checks output and length equality, and reports both forward-only and forward-plus-backward timings. These are isolated operation measurements, not end-to-end FastSpeech model speedups. See [docs/performance.md](docs/performance.md) for the current M1 Max results.

## Benchmark

Run the benchmark against the original PyTorch expression:

```bash
PYTHONPATH=python python benchmarks/benchmark_pair_features.py \
    --batch 1024 --bodies 20 --repeats 30
```

The benchmark checks exact output equality before measuring. It synchronizes MPS around each timed operation. Do not assume a custom kernel is faster: dispatch overhead and memory traffic can outweigh the savings for small tensors.

Initial M1 Max measurements are recorded in [docs/performance.md](docs/performance.md). The feature-building operation was about 4.4× faster in isolation. Full-rollout gains were much smaller and varied by batch size, so the kernels remain opt-in. These are early measurements, not general performance guarantees.

## Using the kernel in ML-phys

Install this project into the ML-phys virtual environment:

```bash
python -m pip install -e /path/to/compute_shaders
```

Then opt in to the custom pair-feature kernel:

```bash
python training/train_interaction.py \
    --device mps \
    --metal-kernels \
    --data data/train_fixed.phdat \
    --out checkpoints/interaction_metal \
    --logdir runs/interaction_metal
```

The option is off by default. CUDA, CPU, and the normal PyTorch MPS path remain unchanged unless explicitly enabled.

The experimental fused pair projection can be tested separately with `--metal-fused-pair-encoder`. It includes a custom backward for state, weight, and bias gradients, but current full-rollout benchmarks do not show a speedup.

## Project status

- [x] Runtime compilation through PyTorch MPS
- [x] Fused pair-feature assembly kernel
- [x] Fused pair projection with custom autograd backward (experimental)
- [x] Python API and input validation
- [x] Correctness tests against the PyTorch reference
- [x] FastSpeech2 length regulation with custom Metal forward/backward kernels
- [x] ML-phys integration behind opt-in switches
- [ ] Faster tiled projection kernel
- [ ] Vulkan backend
- [ ] Wheels, release automation, and broader hardware CI

## License

Apache-2.0. See [LICENSE](LICENSE).
