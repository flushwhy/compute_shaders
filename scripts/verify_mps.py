#!/usr/bin/env python3
"""Compile the Metal source through PyTorch and verify its output."""
import torch
from computeshaders import MPSKernelLibrary

if not torch.backends.mps.is_available():
    raise SystemExit("No MPS device is available.")

kernels = MPSKernelLibrary()
states = torch.randn((2, 5, 12), dtype=torch.float32, device="mps")
actual = kernels.build_pair_features(states)
target = states[:, :, None, :].expand(2, 5, 5, 12)
neighbor = states[:, None, :, :].expand(2, 5, 5, 12)
expected = torch.cat((
    target, neighbor,
    neighbor[..., 0:2] - target[..., 0:2],
    neighbor[..., 3:5] - target[..., 3:5],
), dim=-1)

torch.mps.synchronize()
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
print("Metal kernel verified on Apple MPS.")
print(f"Output shape: {tuple(actual.shape)}")
