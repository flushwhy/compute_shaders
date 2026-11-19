import unittest

import torch

from computeshaders.mps import MPSKernelLibrary, MPSUnavailableError


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


if __name__ == "__main__":
    unittest.main()
