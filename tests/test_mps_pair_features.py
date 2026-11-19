import unittest
from pathlib import Path

import torch

from computeshaders.mps import MPSKernelLibrary


class MPSInputValidationTests(unittest.TestCase):
    def test_packaged_shader_matches_canonical_metal_source(self):
        project_root = Path(__file__).resolve().parents[1]
        canonical = (project_root / "metal" / "pair_features.metal").read_text()
        packaged = (
            project_root / "python" / "computeshaders" / "kernels" / "pair_features.metal"
        ).read_text()
        self.assertEqual(packaged, canonical)

    def test_pair_feature_input_rejects_non_tensor_values(self):
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            MPSKernelLibrary.validate_pair_feature_input(None)

    def test_pair_feature_input_rejects_cpu_tensors(self):
        states = torch.randn((1, 2, 12), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "MPS device"):
            MPSKernelLibrary.validate_pair_feature_input(states)


@unittest.skipUnless(torch.backends.mps.is_available(), "Apple MPS is not available")
class MPSPairFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernels = MPSKernelLibrary()

    def test_pair_features_match_pytorch_reference(self):
        states = torch.randn((2, 5, 12), device="mps", dtype=torch.float32)
        actual = self.kernels.build_pair_features(states)

        target = states[:, :, None, :].expand(2, 5, 5, 12)
        neighbor = states[:, None, :, :].expand(2, 5, 5, 12)
        expected = torch.cat(
            (
                target,
                neighbor,
                neighbor[..., 0:2] - target[..., 0:2],
                neighbor[..., 3:5] - target[..., 3:5],
            ),
            dim=-1,
        )

        torch.mps.synchronize()
        self.assertTrue(torch.equal(actual, expected))

    def test_pair_features_reject_non_contiguous_input(self):
        states = torch.randn((2, 5, 12), device="mps").transpose(0, 1)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.kernels.build_pair_features(states)

    def test_pair_features_reject_inputs_that_require_grad(self):
        states = torch.randn((2, 5, 12), device="mps", requires_grad=True)
        with self.assertRaisesRegex(ValueError, "does not implement backward"):
            self.kernels.build_pair_features(states)

    def test_pair_features_reject_wrong_feature_width(self):
        states = torch.randn((2, 5, 10), device="mps")
        with self.assertRaisesRegex(ValueError, "12 features"):
            self.kernels.build_pair_features(states)

    def test_single_body_produces_one_self_pair(self):
        states = torch.randn((1, 1, 12), device="mps")
        actual = self.kernels.build_pair_features(states)
        torch.mps.synchronize()
        expected = torch.cat(
            (
                states[:, :, None, :],
                states[:, None, :, :],
                torch.zeros((1, 1, 1, 4), device="mps"),
            ),
            dim=-1,
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_threadgroup_size_must_be_a_valid_multiple_of_32(self):
        with self.assertRaisesRegex(ValueError, "multiple of 32"):
            MPSKernelLibrary(group_size=48)

    def test_fused_pair_projection_matches_linear_reference_and_all_gradients(self):
        torch.manual_seed(27)
        states = torch.randn((2, 4, 12), device="mps", requires_grad=True)
        weight = torch.randn((16, 28), device="mps", requires_grad=True)
        bias = torch.randn((16,), device="mps", requires_grad=True)

        target = states[:, :, None, :].expand(2, 4, 4, 12)
        neighbor = states[:, None, :, :].expand(2, 4, 4, 12)
        features = torch.cat(
            (
                target,
                neighbor,
                neighbor[..., 0:2] - target[..., 0:2],
                neighbor[..., 3:5] - target[..., 3:5],
            ),
            dim=-1,
        )
        expected = torch.nn.functional.linear(features, weight, bias)
        expected.square().mean().backward()
        expected_grads = (states.grad.clone(), weight.grad.clone(), bias.grad.clone())

        states.grad = None
        weight.grad = None
        bias.grad = None
        actual = self.kernels.project_pair_features(states, weight, bias)
        actual.square().mean().backward()
        torch.mps.synchronize()

        torch.testing.assert_close(actual, expected.detach(), rtol=1e-5, atol=1e-5)
        for actual_grad, expected_grad in zip(
            (states.grad, weight.grad, bias.grad), expected_grads
        ):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
