"""FastSpeech-style length regulation using custom Apple Metal kernels."""

from __future__ import annotations

from pathlib import Path
from typing import Any


_METAL_GROUP_SIZE = 256


class MPSLengthRegulatorUnavailableError(RuntimeError):
    """Raised when custom MPS length regulation cannot run on this machine."""


def _make_length_regulator_function(torch: Any, library: Any) -> Any:
    class LengthRegulate(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any,
            states: Any,
            exclusive_cumsum: Any,
            mel_lengths: Any,
            output_length: int,
        ) -> Any:
            batch_size, source_length, hidden_size = states.shape
            output = torch.empty(
                (batch_size, output_length, hidden_size),
                dtype=states.dtype,
                device=states.device,
            )

            if output_length:
                library.length_regulate(
                    states,
                    exclusive_cumsum,
                    mel_lengths,
                    output,
                    source_length,
                    hidden_size,
                    output_length,
                    batch_size,
                    threads=[batch_size * output_length * hidden_size, 1, 1],
                    group_size=[_METAL_GROUP_SIZE, 1, 1],
                )

            ctx.library = library
            ctx.source_shape = states.shape
            ctx.output_length = output_length
            ctx.save_for_backward(exclusive_cumsum, mel_lengths)
            return output

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, None, None, None]:
            exclusive_cumsum, mel_lengths = ctx.saved_tensors
            batch_size, source_length, hidden_size = ctx.source_shape
            grad_states = torch.empty(
                (batch_size, source_length, hidden_size),
                dtype=grad_output.dtype,
                device=grad_output.device,
            )

            if ctx.output_length:
                ctx.library.length_regulate_backward(
                    grad_output.contiguous(),
                    exclusive_cumsum,
                    mel_lengths,
                    grad_states,
                    source_length,
                    hidden_size,
                    ctx.output_length,
                    batch_size,
                    threads=[batch_size * source_length * hidden_size, 1, 1],
                    group_size=[_METAL_GROUP_SIZE, 1, 1],
                )
            else:
                grad_states.zero_()

            return grad_states, None, None, None

    return LengthRegulate


class MPSLengthRegulator:
    """Repeat phoneme states by duration without per-phoneme CPU synchronization.

    Inputs and outputs stay on MPS. If max_len is omitted, determining the output
    tensor width requires one scalar synchronization. Pass max_len from the
    batch when it is already known to avoid that synchronization.
    """

    def __init__(self, source_path: str | Path | None = None) -> None:
        try:
            import torch
        except ImportError as error:
            raise MPSLengthRegulatorUnavailableError(
                "PyTorch is required. Install the MPS extra with torch>=2.7."
            ) from error

        if not hasattr(torch.mps, "compile_shader"):
            raise MPSLengthRegulatorUnavailableError(
                "torch.mps.compile_shader is unavailable; install PyTorch 2.7 or newer."
            )
        if not torch.backends.mps.is_available():
            raise MPSLengthRegulatorUnavailableError(
                "An available Apple MPS device is required."
            )

        shader_path = Path(source_path) if source_path else (
            Path(__file__).resolve().parent / "kernels" / "length_regulator.metal"
        )
        if not shader_path.is_file():
            raise FileNotFoundError(f"Metal shader source not found: {shader_path}")

        self._torch = torch
        shader_source = shader_path.read_text(encoding="utf-8")
        self._library = torch.mps.compile_shader(shader_source)
        self._operation = _make_length_regulator_function(torch, self._library)

    def __call__(
        self,
        states: Any,
        durations: Any,
        max_len: int | None = None,
    ) -> tuple[Any, Any]:
        return self.forward(states, durations, max_len)

    def forward(
        self,
        states: Any,
        durations: Any,
        max_len: int | None = None,
    ) -> tuple[Any, Any]:
        torch = self._torch

        if not isinstance(states, torch.Tensor) or not isinstance(durations, torch.Tensor):
            raise TypeError("states and durations must be torch.Tensor values.")
        if states.device.type != "mps" or durations.device.type != "mps":
            raise ValueError("states and durations must both be on the MPS device.")
        if states.dtype != torch.float32:
            raise TypeError("states must use torch.float32.")
        if durations.dtype not in (torch.int32, torch.int64):
            raise TypeError("durations must use torch.int32 or torch.int64.")
        if states.ndim != 3:
            raise ValueError("states must have shape (batch, source_length, hidden_size).")
        if durations.ndim != 2 or tuple(durations.shape) != tuple(states.shape[:2]):
            raise ValueError("durations must have shape (batch, source_length).")
        if min(states.shape) < 1:
            raise ValueError("states dimensions must all be non-zero.")
        if durations.requires_grad:
            raise ValueError("durations cannot require gradients.")
        if max_len is not None and (
            isinstance(max_len, bool) or not isinstance(max_len, int) or max_len < 0
        ):
            raise ValueError("max_len must be a non-negative integer or None.")

        states = states.contiguous()
        nonnegative_durations = durations.to(dtype=torch.int32).clamp_min(0).contiguous()

        # Metal reads int32; request that dtype because cumsum otherwise
        # promotes integer inputs to int64 by default.
        inclusive_cumsum = torch.cumsum(
            nonnegative_durations,
            dim=1,
            dtype=torch.int32,
        )
        exclusive_cumsum = (inclusive_cumsum - nonnegative_durations).contiguous()
        mel_lengths = inclusive_cumsum[:, -1].contiguous()

        output_length = (
            int(mel_lengths.max().item()) if max_len is None else max_len
        )
        batch_size, source_length, hidden_size = states.shape
        output_values = batch_size * output_length * hidden_size
        input_values = batch_size * source_length * hidden_size
        if max(output_values, input_values) > 0x7FFFFFFF:
            raise ValueError("The tensor exceeds the supported Metal indexing range.")

        output = self._operation.apply(
            states,
            exclusive_cumsum,
            mel_lengths,
            output_length,
        )
        return output, mel_lengths.to(dtype=torch.int64)
