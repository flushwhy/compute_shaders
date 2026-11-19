#!/usr/bin/env python3
"""Compile and verify the custom Metal kernels through PyTorch MPS."""
import torch
from computeshaders import MPSKernelLibrary

if not torch.backends.mps.is_available():
    print("SKIP: this runner does not expose a Metal GPU; run on Apple Silicon to verify the shader.")
    raise SystemExit(0)

kernels = MPSKernelLibrary()
states = torch.randn((2, 5, 12), dtype=torch.float32, device="mps")
actual = kernels.build_pair_features(states)
target = states[:, :, None, :].expand(2, 5, 5, 12)
neighbor = states[:, None, :, :].expand(2, 5, 5, 12)
expected = torch.cat((
    target,
    neighbor,
    neighbor[..., 0:2] - target[..., 0:2],
    neighbor[..., 3:5] - target[..., 3:5],
), dim=-1)

torch.testing.assert_close(actual, expected, rtol=0, atol=0)
print("Pair-feature Metal kernel verified.")

projection_states = torch.randn((2, 5, 12), dtype=torch.float32, device="mps", requires_grad=True)
projection_weight = torch.randn((16, 28), dtype=torch.float32, device="mps", requires_grad=True)
projection_bias = torch.randn((16,), dtype=torch.float32, device="mps", requires_grad=True)
target = projection_states[:, :, None, :].expand(2, 5, 5, 12)
neighbor = projection_states[:, None, :, :].expand(2, 5, 5, 12)
pair_features = torch.cat((
    target,
    neighbor,
    neighbor[..., 0:2] - target[..., 0:2],
    neighbor[..., 3:5] - target[..., 3:5],
), dim=-1)
reference = torch.nn.functional.linear(pair_features, projection_weight, projection_bias)
reference.square().mean().backward()
reference_grads = (
    projection_states.grad.clone(),
    projection_weight.grad.clone(),
    projection_bias.grad.clone(),
)

projection_states.grad = None
projection_weight.grad = None
projection_bias.grad = None
fused = kernels.project_pair_features(
    projection_states, projection_weight, projection_bias
)
fused.square().mean().backward()
torch.mps.synchronize()
torch.testing.assert_close(fused, reference.detach(), rtol=1e-5, atol=1e-5)
for actual_grad, expected_grad in zip(
    (projection_states.grad, projection_weight.grad, projection_bias.grad),
    reference_grads,
):
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-5)

print("Fused pair-projection Metal kernel and custom autograd verified.")
print(f"Fused output shape: {tuple(fused.shape)}")
