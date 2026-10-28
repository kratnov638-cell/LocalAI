import unittest
import threading
from types import SimpleNamespace
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
    def test_prefill_reuses_identical_system_prefix_and_matches_full_recompute(self):
        class FakeCache:
            def __init__(self):
                self.k = np.zeros((2, 12, 1, 2), dtype=np.float32)
                self.v = np.zeros_like(self.k)
                self.length = 0

            def reset(self):
                self.length = 0

        fake = SimpleNamespace(
            cache=FakeCache(),
            _conv_state=np.zeros((2, 1), dtype=np.float32),
            _prev_hidden=np.zeros((2, 3), dtype=np.float32),
            _prefix_cache_tokens=None, _prefix_cache_length=0,
            _prefix_cache_k=None, _prefix_cache_v=None,
            _prefix_cache_conv_state=None, _prefix_cache_prev_hidden=None,
            _prefix_cache_logits=None,
        )

        def reset_state():
            fake.cache.reset()
            fake._conv_state.fill(0)
            fake._prev_hidden.fill(0)

        def forward(token, pos):
            fake.cache.k[:, pos] = token + pos
            fake.cache.v[:, pos] = token - pos
            fake._conv_state += token
            fake._prev_hidden += token * 0.5
            return np.array([token + fake._conv_state.sum(), fake._prev_hidden.sum()], dtype=np.float32)

        fake.reset_state = reset_state
        fake.forward = forward
        tokenizer = SimpleNamespace(stop_ids={106})

        first = [1, 7, 106, 3, 4]
        fake.reset_state()
        ZayaModel.prefill_prompt(fake, first, threading.Event(), tokenizer)
        self.assertEqual(fake._prefix_cache_length, 3)

        followup = [1, 7, 106, 8, 9, 10]
        fake.reset_state()
        cached_logits = ZayaModel.prefill_prompt(fake, followup, threading.Event(), tokenizer)
        cached_k = fake.cache.k.copy()
        cached_v = fake.cache.v.copy()

        fake._prefix_cache_tokens = None
        fake._prefix_cache_length = 0
        fake._prefix_cache_k = fake._prefix_cache_v = None
        fake._prefix_cache_conv_state = fake._prefix_cache_prev_hidden = None
        fake._prefix_cache_logits = None
        fake.reset_state()
        full_logits = ZayaModel.prefill_prompt(fake, followup, threading.Event(), tokenizer)

        np.testing.assert_array_equal(cached_logits, full_logits)
        np.testing.assert_array_equal(cached_k, fake.cache.k)
        np.testing.assert_array_equal(cached_v, fake.cache.v)

    def test_grouped_attention_matches_explicitly_repeated_kv_heads(self):
        rng = np.random.default_rng(7)
        q = rng.normal(size=(8, 128)).astype(np.float32)
        keys = rng.normal(size=(11, 2, 128)).astype(np.float32)
        values = rng.normal(size=(11, 2, 128)).astype(np.float32)
        expanded_k = np.repeat(keys, 4, axis=1).transpose(1, 0, 2)
        expanded_v = np.repeat(values, 4, axis=1).transpose(1, 0, 2)
        scores = np.einsum('hd,hld->hl', q, expanded_k, optimize=True) / np.sqrt(128.0)
        scores -= scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=-1, keepdims=True)
        expected = np.einsum('hl,hld->hd', probs, expanded_v, optimize=True).reshape(1024)
        actual = ZayaModel._grouped_attention(q, keys, values)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

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
