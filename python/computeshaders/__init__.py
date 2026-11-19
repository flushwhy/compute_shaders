"""Reusable custom compute kernels for Apple Silicon and PyTorch MPS."""

from .mps import MPSKernelLibrary, MPSUnavailableError

__all__ = ["MPSKernelLibrary", "MPSUnavailableError"]
__version__ = "0.1.0"
