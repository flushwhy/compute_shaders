"""Compare the fused Metal kernel with the existing PyTorch expression."""
import argparse
import statistics
import time

import torch
from computeshaders import MPSKernelLibrary


def reference(states):
    batch, bodies, _ = states.shape
    target = states[:, :, None, :].expand(batch, bodies, bodies, 12)
    neighbor = states[:, None, :, :].expand(batch, bodies, bodies, 12)
    return torch.cat((
        target, neighbor,
        neighbor[..., 0:2] - target[..., 0:2],
        neighbor[..., 3:5] - target[..., 3:5],
    ), dim=-1)


def measure(operation, repeats):
    samples = []
    for _ in range(repeats):
        torch.mps.synchronize()
        start = time.perf_counter()
        result = operation()
        torch.mps.synchronize()
        samples.append(time.perf_counter() - start)
        del result
    return statistics.median(samples) * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--bodies", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--group-size", type=int, default=256)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available.")

    states = torch.randn((args.batch, args.bodies, 12), device="mps")
    kernels = MPSKernelLibrary(group_size=args.group_size)
    custom = kernels.build_pair_features(states)
    baseline = reference(states)
    torch.mps.synchronize()
    torch.testing.assert_close(custom, baseline, rtol=0, atol=0)

    baseline_ms = measure(lambda: reference(states), args.repeats)
    custom_ms = measure(lambda: kernels.build_pair_features(states), args.repeats)
    print(f"Shape:              ({args.batch}, {args.bodies}, 12)")
    print(f"PyTorch reference:  {baseline_ms:.3f} ms")
    print(f"Custom Metal:       {custom_ms:.3f} ms")
    print(f"Threadgroup size:   {args.group_size}")
    print(f"Speedup:            {baseline_ms / custom_ms:.2f}x")


if __name__ == "__main__":
    main()
