from __future__ import annotations

import argparse
import platform
import statistics
import subprocess
import time
from collections.abc import Callable

import torch

from computeshaders import MPSLengthRegulator


Operation = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def original_loop(
    states: torch.Tensor,
    durations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation with the original per-item/per-phoneme loops."""
    sequences = []
    lengths = []

    for batch_states, batch_durations in zip(states, durations):
        repeated = []
        for state, duration in zip(batch_states, batch_durations):
            count = max(int(duration.item()), 0)
            if count:
                repeated.append(state.unsqueeze(0).expand(count, -1))

        if repeated:
            sequence = torch.cat(repeated, dim=0)
        else:
            sequence = states.new_empty((0, states.shape[-1]))

        sequences.append(sequence)
        lengths.append(sequence.shape[0])

    output_width = max(lengths)
    output = states.new_zeros((states.shape[0], output_width, states.shape[-1]))
    for batch, sequence in enumerate(sequences):
        output[batch, :sequence.shape[0]] = sequence

    mel_lengths = torch.tensor(lengths, dtype=torch.long, device=states.device)
    return output, mel_lengths


def make_training_step(operation: Operation) -> Callable:
    def run(states: torch.Tensor, durations: torch.Tensor) -> None:
        states.grad = None
        output, _ = operation(states, durations)
        output.sum().backward()

    return run


def measure(
    function: Callable,
    states: torch.Tensor,
    durations: torch.Tensor,
    warmup: int,
    repetitions: int,
) -> float:
    for _ in range(warmup):
        function(states, durations)
    torch.mps.synchronize()

    samples = []
    for _ in range(repetitions):
        torch.mps.synchronize()
        start = time.perf_counter()
        function(states, durations)
        torch.mps.synchronize()
        samples.append((time.perf_counter() - start) * 1000)

    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the original FastSpeech2 loop with the custom Metal kernel."
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("This benchmark requires an available Apple MPS device.")

    device = torch.device("mps")
    length_regulator = MPSLengthRegulator()
    configs = [
        ("small", 2, 30, 256),
        ("medium", 8, 100, 256),
        ("large", 16, 200, 256),
    ]

    chip = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip() or platform.processor() or platform.machine()

    print(f"PyTorch {torch.__version__} | chip: {chip} | macOS: {platform.mac_ver()[0]}")
    print(f"warmup={args.warmup} repetitions={args.repetitions}")
    print(
        f"{'case':<10} {'B':>4} {'S':>5} {'H':>5} "
        f"{'loop fwd':>11} {'Metal fwd':>11} {'speedup':>9} "
        f"{'loop train':>11} {'Metal train':>12} {'speedup':>9}"
    )

    for label, batch, source_length, hidden_size in configs:
        states = torch.randn(batch, source_length, hidden_size, device=device)
        durations = torch.randint(
            1,
            6,
            (batch, source_length),
            dtype=torch.int32,
            device=device,
        )

        expected, expected_lengths = original_loop(states, durations)
        actual, actual_lengths = length_regulator(states, durations)
        torch.mps.synchronize()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_lengths, expected_lengths)

        loop_forward_ms = measure(
            original_loop,
            states,
            durations,
            args.warmup,
            args.repetitions,
        )
        metal_forward_ms = measure(
            length_regulator,
            states,
            durations,
            args.warmup,
            args.repetitions,
        )

        training_states = states.detach().clone().requires_grad_(True)
        loop_training_ms = measure(
            make_training_step(original_loop),
            training_states,
            durations,
            args.warmup,
            args.repetitions,
        )
        metal_training_ms = measure(
            make_training_step(length_regulator),
            training_states,
            durations,
            args.warmup,
            args.repetitions,
        )

        print(
            f"{label:<10} {batch:>4} {source_length:>5} {hidden_size:>5} "
            f"{loop_forward_ms:>10.3f} {metal_forward_ms:>10.3f} "
            f"{loop_forward_ms / metal_forward_ms:>8.2f}x "
            f"{loop_training_ms:>10.3f} {metal_training_ms:>11.3f} "
            f"{loop_training_ms / metal_training_ms:>8.2f}x"
        )


if __name__ == "__main__":
    main()
