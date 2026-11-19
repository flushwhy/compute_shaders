"""Custom Metal kernels that operate directly on PyTorch MPS tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MPSUnavailableError(RuntimeError):
    """Raised when PyTorch or the current machine cannot run custom MPS kernels."""


class MPSKernelLibrary:
    """Compile and invoke the project's Metal kernels on PyTorch MPS tensors.

    Kernels are compiled once when the library is created. The API deliberately
    accepts and returns MPS tensors, avoiding host copies in the hot path.
    """

    def __init__(
        self,
        source_path: str | Path | None = None,
        group_size: int = 256,
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise MPSUnavailableError(
                "PyTorch is required. Install a PyTorch build with MPS support."
            ) from error

        if not hasattr(torch.mps, "compile_shader"):
            raise MPSUnavailableError(
                "This PyTorch build does not expose torch.mps.compile_shader; "
                "install PyTorch 2.7 or newer."
            )
        if not torch.backends.mps.is_available():
            raise MPSUnavailableError("An available Apple MPS device is required.")
        if not isinstance(group_size, int) or group_size < 32 or group_size > 1024 or group_size % 32:
            raise ValueError("group_size must be a multiple of 32 between 32 and 1024.")

        shader_path = Path(source_path) if source_path else (
            Path(__file__).resolve().parent / "kernels" / "pair_features.metal"
        )
        if not shader_path.is_file():
            raise FileNotFoundError(f"Metal shader source not found: {shader_path}")

        self._torch = torch
        self.group_size = group_size
        self._library = torch.mps.compile_shader(shader_path.read_text(encoding="utf-8"))

    @staticmethod
    def validate_pair_feature_input(states: Any) -> None:
        import torch

        if not isinstance(states, torch.Tensor):
            raise TypeError("states must be a torch.Tensor.")
        if states.device.type != "mps":
            raise ValueError("states must be on the MPS device.")
        if states.dtype != torch.float32:
            raise TypeError("states must use torch.float32.")
        if states.ndim != 3 or states.shape[-1] != 12:
            raise ValueError("states must have shape (batch, bodies, 12 features).")
        if states.shape[0] < 1 or states.shape[1] < 1:
            raise ValueError("states must contain at least one batch and one body.")
        if not states.is_contiguous():
            raise ValueError("states must be contiguous.")
        if states.requires_grad:
            raise ValueError(
                "This input-feature kernel does not implement backward; "
                "states must not require gradients."
            )
        if states.shape[0] * states.shape[1] * states.shape[1] > 0x7FFFFFFF:
            raise ValueError("The pair count exceeds the supported Metal dispatch size.")

    def build_pair_features(self, states: Any) -> Any:
        """Build the model's 28-value pair features from (B, N, 12) states.

        Output layout is (B, N, N, 28): target state, neighbor state, then
        relative x/y position and velocity. This matches ML-phys exactly.
        """
        self.validate_pair_feature_input(states)
        torch = self._torch
        batch, bodies, _ = states.shape
        output = torch.empty(
            (batch, bodies, bodies, 28),
            dtype=torch.float32,
            device=states.device,
        )
        total_pairs = batch * bodies * bodies
        self._library.build_pair_features(
            states,
            output,
            bodies,
            threads=[total_pairs, 1, 1],
            group_size=[self.group_size, 1, 1],
        )
        return output
