"""Custom Metal kernels that operate directly on PyTorch MPS tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MPSUnavailableError(RuntimeError):
    """Raised when PyTorch or the current machine cannot run custom MPS kernels."""


def _make_fused_projection_function(torch: Any, library: Any) -> Any:
    """Create a custom autograd operation around the fused Metal projection."""

    class FusedPairProjection(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, states: Any, weight: Any, bias: Any) -> Any:
            batch, bodies, _ = states.shape
            output_features = weight.shape[0]
            output = torch.empty(
                (batch, bodies, bodies, output_features),
                dtype=states.dtype,
                device=states.device,
            )
            total_values = batch * bodies * bodies * 32
            library._library.project_pair_features(
                states,
                weight,
                bias,
                output,
                bodies,
                output_features,
                batch,
                threads=[total_values, 1, 1],
                group_size=[library.group_size, 1, 1],
            )
            ctx.library = library
            ctx.save_for_backward(states, weight)
            return output

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, Any, Any]:
            states, weight = ctx.saved_tensors
            grad_states = grad_weight = grad_bias = None
            grad_output = grad_output.contiguous()

            if ctx.needs_input_grad[1]:
                pair_features = ctx.library.build_pair_features(states.detach())
                grad_flat = grad_output.reshape(-1, weight.shape[0])
                feature_flat = pair_features.reshape(-1, 28)
                grad_weight = grad_flat.transpose(0, 1).matmul(feature_flat)

            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(dim=(0, 1, 2))

            if ctx.needs_input_grad[0]:
                grad_features = torch.matmul(grad_output, weight)
                direct_target = grad_features[..., 0:12].sum(dim=2)
                direct_neighbor = grad_features[..., 12:24].sum(dim=1)
                relative = grad_features[..., 24:28]
                relative_gradient = relative.sum(dim=1) - relative.sum(dim=2)
                grad_states = torch.cat(
                    (
                        direct_target[..., 0:2]
                        + direct_neighbor[..., 0:2]
                        + relative_gradient[..., 0:2],
                        direct_target[..., 2:3] + direct_neighbor[..., 2:3],
                        direct_target[..., 3:5]
                        + direct_neighbor[..., 3:5]
                        + relative_gradient[..., 2:4],
                        direct_target[..., 5:12] + direct_neighbor[..., 5:12],
                    ),
                    dim=-1,
                )

            return grad_states, grad_weight, grad_bias

    return FusedPairProjection


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
        self._fused_projection = _make_fused_projection_function(torch, self)

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

    def project_pair_features(self, states: Any, weight: Any, bias: Any) -> Any:
        """Fuse 28-value pair construction and a Linear projection.

        The custom autograd backward recomputes pair features and supplies
        gradients for the input states, projection weights, and bias. This
        avoids storing the large pair-feature tensor during the forward pass.
        """
        torch = self._torch
        if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
            raise TypeError("weight and bias must be torch.Tensor values.")
        if states.device.type != "mps" or weight.device.type != "mps" or bias.device.type != "mps":
            raise ValueError("states, weight, and bias must be on the MPS device.")
        if states.dtype != torch.float32 or weight.dtype != torch.float32 or bias.dtype != torch.float32:
            raise TypeError("states, weight, and bias must use torch.float32.")
        if states.ndim != 3 or states.shape[-1] != 12 or states.shape[0] < 1 or states.shape[1] < 1:
            raise ValueError("states must have shape (batch, bodies, 12 features).")
        if not states.is_contiguous():
            raise ValueError("states must be contiguous.")
        if weight.ndim != 2 or weight.shape[1] != 28 or weight.shape[0] < 1:
            raise ValueError("weight must have shape (output_features, 28).")
        if bias.shape != (weight.shape[0],):
            raise ValueError("bias must match the projection output width.")
        if not weight.is_contiguous() or not bias.is_contiguous():
            raise ValueError("weight and bias must be contiguous.")
        total_pairs = states.shape[0] * states.shape[1] * states.shape[1]
        if total_pairs * 32 > 0x7FFFFFFF:
            raise ValueError("The projection dispatch exceeds the supported Metal grid size.")
        if total_pairs * weight.shape[0] > 0x7FFFFFFF:
            raise ValueError("The projection output exceeds the supported Metal indexing range.")

        return self._fused_projection.apply(states, weight, bias)

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
