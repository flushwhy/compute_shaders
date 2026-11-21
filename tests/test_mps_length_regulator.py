from __future__ import annotations

import unittest

import torch

from computeshaders.length_regulator import MPSLengthRegulator


def reference_length_regulate(
    states: torch.Tensor,
    durations: torch.Tensor,
    max_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded = []
    lengths = []

    for batch_states, batch_durations in zip(states, durations):
        frames = []
        for state, duration in zip(batch_states, batch_durations):
            count = max(int(duration.item()), 0)
            frames.append(state.unsqueeze(0).expand(count, -1))

        sequence = torch.cat(frames, dim=0)
        expanded.append(sequence)
        lengths.append(sequence.shape[0])

    output_width = max_len if max_len is not None else max(lengths)
    output = states.new_zeros((states.shape[0], output_width, states.shape[2]))
    for batch, sequence in enumerate(expanded):
        output[batch, : min(sequence.shape[0], output_width)] = sequence[:output_width]

    return output, torch.tensor(lengths, dtype=torch.long, device=states.device)


@unittest.skipUnless(torch.backends.mps.is_available(), "Apple MPS is not available")
class TestMPSLengthRegulator(unittest.TestCase):
    def test_matches_reference_with_zero_durations(self) -> None:
        states = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3).to("mps")
        durations = torch.tensor(
            [[2, 0, 1, 3], [0, 2, 0, 1]],
            dtype=torch.int32,
            device="mps",
        )

        actual, actual_lengths = MPSLengthRegulator()(states, durations)
        expected, expected_lengths = reference_length_regulate(states, durations)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_lengths, expected_lengths)

    def test_preserves_gradients(self) -> None:
        states = torch.randn(2, 4, 5, device="mps", requires_grad=True)
        durations = torch.tensor(
            [[2, 0, 1, 3], [1, 2, 0, 1]],
            dtype=torch.int32,
            device="mps",
        )

        actual, _ = MPSLengthRegulator()(states, durations)
        output_gradient = torch.randn_like(actual)
        actual.backward(output_gradient)

        expected_states = states.detach().clone().requires_grad_(True)
        expected, _ = reference_length_regulate(expected_states, durations)
        expected.backward(output_gradient)

        torch.testing.assert_close(states.grad, expected_states.grad)

    def test_accepts_explicit_output_width(self) -> None:
        states = torch.randn(2, 3, 4, device="mps")
        durations = torch.tensor(
            [[1, 2, 1], [2, 1, 1]],
            dtype=torch.int64,
            device="mps",
        )

        actual, lengths = MPSLengthRegulator()(states, durations, max_len=5)

        self.assertEqual(tuple(actual.shape), (2, 5, 4))
        torch.testing.assert_close(lengths, torch.tensor([4, 4], device="mps"))

    def test_rejects_wrong_device(self) -> None:
        states = torch.randn(1, 2, 3)
        durations = torch.ones(1, 2, dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "MPS"):
            MPSLengthRegulator()(states, durations)

    def test_clamps_negative_durations(self) -> None:
        states = torch.randn(1, 3, 4, device="mps")
        durations = torch.tensor([[-2, 2, 1]], dtype=torch.int32, device="mps")

        actual, lengths = MPSLengthRegulator()(states, durations)
        expected, expected_lengths = reference_length_regulate(states, durations)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(lengths, expected_lengths)

    def test_handles_all_zero_durations(self) -> None:
        states = torch.randn(2, 3, 4, device="mps", requires_grad=True)
        durations = torch.zeros(2, 3, dtype=torch.int32, device="mps")

        actual, lengths = MPSLengthRegulator()(states, durations)

        self.assertEqual(tuple(actual.shape), (2, 0, 4))
        torch.testing.assert_close(lengths, torch.zeros(2, dtype=torch.long, device="mps"))
        actual.sum().backward()
        torch.testing.assert_close(states.grad, torch.zeros_like(states))

    def test_accepts_noncontiguous_inputs(self) -> None:
        states = torch.randn(2, 5, 4, device="mps").transpose(1, 2)
        durations = torch.tensor(
            [[1, 2], [3, 1], [2, 0], [1, 2]],
            dtype=torch.int32,
            device="mps",
        ).transpose(0, 1)

        actual, actual_lengths = MPSLengthRegulator()(states, durations)
        expected, expected_lengths = reference_length_regulate(states, durations)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_lengths, expected_lengths)


if __name__ == "__main__":
    unittest.main()
