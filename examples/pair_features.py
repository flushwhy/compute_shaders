"""Run the pair-feature Metal kernel on a small MPS tensor."""
import torch
from computeshaders import MPSKernelLibrary

if not torch.backends.mps.is_available():
    raise SystemExit("This example requires an Apple Silicon Mac with MPS enabled.")

states = torch.randn((4, 20, 12), device="mps", dtype=torch.float32)
kernels = MPSKernelLibrary()
pair_features = kernels.build_pair_features(states)
torch.mps.synchronize()

print(f"Input:  {tuple(states.shape)}")
print(f"Output: {tuple(pair_features.shape)}")
print(f"Device: {pair_features.device}")
