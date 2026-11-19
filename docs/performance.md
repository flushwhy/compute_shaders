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

| Workload | PyTorch | Custom Metal | Speedup |
|---|---:|---:|---:|
| Full model forward/backward, batch 1,024 | 75.847 ms | 75.847 ms | 1.00× |
| Three-step rollout forward/backward, batch 256 | 111.012 ms | 102.397 ms | 1.08× |

The three-step rollout used synthetic states and targets, 20 bodies, and 12 paired synchronized samples after warm-up. It is closer to the training loop than the isolated kernel benchmark, but it is still not a full dataset training run.

This is the important qualification: accelerating feature assembly does not automatically accelerate the full model. The pair encoder and the rest of the training graph dominate enough of the runtime that a single forward/backward pass showed no measurable difference. The rollout benchmark showed a modest improvement of about 8%.

The custom path therefore remains opt-in. More useful gains likely require fusing feature generation with the first pair-encoder projection, while preserving gradients for the model weights.

## Reproducing

```bash
PYTHONPATH=python python benchmarks/benchmark_pair_features.py \
    --batch 1024 --bodies 20 --repeats 30 --group-size 256
```

Always rerun on the target machine and PyTorch version. Different Apple GPU generations, batch sizes, and body counts can change the best threadgroup size.
