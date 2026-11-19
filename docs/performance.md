# Initial performance notes

These are development measurements, not promises about every Apple GPU.

## M1 Max development machine

- macOS: 26.5.2
- PyTorch: 2.7.0
- dtype: float32
- tensor shape: batch 1,024 × 20 bodies × 12 state values
- output: batch 1,024 × 20 × 20 pairs × 28 values
- measurement: median of synchronized MPS timings after warm-up

| Implementation | Median | Relative |
|---|---:|---:|
| PyTorch reference feature assembly | 1.494 ms | 1.00× |
| Custom Metal, 64 threads/group | 0.389 ms | 3.84× |
| Custom Metal, 128 threads/group | 0.349 ms | 4.28× |
| Custom Metal, 256 threads/group | 0.341 ms | 4.38× |
| Custom Metal, 512 threads/group | 0.347 ms | 4.30× |

The vectorized float4 kernel with a 256-thread group was the best of these tested configurations.

## Model-level measurements

The model-level numbers below were rerun with the fused projection implementation. Each measurement is a median of synchronized forward/backward samples after warm-up, using synthetic states and targets with 20 bodies, hidden width 128, and pair width 96.

| Workload | PyTorch | Metal features only | Fused Metal projection |
|---|---:|---:|---:|
| Three-step rollout, batch 256 | 107.72 ms | 107.76 ms | 111.97 ms |
| Three-step rollout, batch 1,024 | 258.12 ms | 253.59 ms | 265.08 ms |

At batch 1,024 the feature-only kernel was about 1.8% faster in this run. The fused projection was about 2.6% slower. At batch 256, both custom paths were within a few percent of the reference. These small differences are close enough that they should be treated as workload- and run-dependent, not as guaranteed gains.

The fused kernel is still useful as an experimental implementation and as a way to avoid materializing the 28-value pair tensor during the forward pass. Its custom autograd backward recomputes those features and returns gradients for states, weights, and bias. That backward recomputation currently costs enough to erase the expected speedup.

For now, use `--metal-kernels` to test the feature-only kernel. Keep `--metal-fused-pair-encoder` experimental until a better tiled matrix implementation improves full-rollout throughput.

## Reproducing

```bash
PYTHONPATH=python python benchmarks/benchmark_pair_features.py \
    --batch 1024 --bodies 20 --repeats 30 --group-size 256
```

Always rerun on the target machine and PyTorch version. Different Apple GPU generations, batch sizes, and body counts can change the best threadgroup size.
