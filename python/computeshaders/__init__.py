"""Reusable custom compute kernels for Apple Silicon and PyTorch MPS."""

from .length_regulator import MPSLengthRegulator, MPSLengthRegulatorUnavailableError
from .mps import MPSKernelLibrary, MPSUnavailableError

__all__ = [
    "MPSKernelLibrary",
    "MPSLengthRegulator",
    "MPSLengthRegulatorUnavailableError",
    "MPSUnavailableError",
]
__version__ = "0.1.0"
