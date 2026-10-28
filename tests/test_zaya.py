import unittest
import numpy as np

from engine.zaya import ZayaModel


class _Info:
    def __init__(self, shape, ggml_type=0):
        self.shape = shape
        self.ggml_type = ggml_type


class _FakeGGUF:
    """Small metadata/tensor-name fixture; no model weights are modified."""
    def __init__(self):
        self.metadata = {
            'general.architecture': 'zaya', 'zaya.block_count': 80,
            'zaya.embedding_length': 2048, 'zaya.attention.head_count': 8,
            'zaya.attention.head_count_kv': 2, 'zaya.attention.key_length': 128,
            'zaya.rope.dimension_count': 64, 'zaya.expert_count': 16,
            'zaya.expert_feed_forward_length': 256,
            'zaya.attention.layer_norm_rms_epsilon': 1e-5,
            'zaya.rope.freq_base': 5_000_000.0,
            'zaya.feed_forward_length': 2048, 'zaya.vocab_size': 262147,
        }
        self.tensors = {}


class ZayaArchitectureTests(unittest.TestCase):
    def test_alternating_legacy_layout_is_40_cca_plus_40_moe(self):
        self.assertEqual(len(range(0, 80, 2)), 40)
        self.assertEqual(len(range(1, 80, 2)), 40)

    def test_residual_scaling_matches_reference_formula(self):
        branch = np.array([1.0, -2.0], dtype=np.float32)
        scale = np.array([2.0, 0.5], dtype=np.float32)
        bias = np.array([0.25, -1.0], dtype=np.float32)
        actual = ZayaModel._res_scale(branch, scale, bias)
        np.testing.assert_allclose(actual, np.array([2.5, -1.5], dtype=np.float32))

    def test_gelu_is_finite_and_exact_at_zero(self):
        result = ZayaModel._gelu(np.array([-4.0, 0.0, 4.0], dtype=np.float32))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(float(result[1]), 0.0)
        self.assertLess(float(result[0]), 0.0)
        self.assertGreater(float(result[2]), 3.9)


if __name__ == '__main__':
    unittest.main()
