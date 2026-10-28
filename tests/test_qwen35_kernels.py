import unittest

import numpy as np

from engine.numba_kernels import delta_net_step


class Qwen35KernelTests(unittest.TestCase):
    def test_fused_delta_net_step_matches_numpy_reference(self):
        rng = np.random.default_rng(3501)
        heads, key_dim, value_dim = 4, 16, 12
        initial = (rng.standard_normal((heads, key_dim, value_dim)) * 0.05).astype(np.float32)
        q = rng.standard_normal((heads, key_dim)).astype(np.float32) / np.sqrt(key_dim)
        k = rng.standard_normal((heads, key_dim)).astype(np.float32)
        v = rng.standard_normal((heads, value_dim)).astype(np.float32)
        decay = (-rng.random(heads)).astype(np.float32)
        beta = rng.random(heads).astype(np.float32)

        expected_state = initial.copy()
        expected_state *= np.exp(decay)[:, None, None]
        predicted = np.einsum('hkv,hk->hv', expected_state, k, optimize=True)
        delta = (v - predicted) * beta[:, None]
        expected_state += k[:, :, None] * delta[:, None, :]
        expected_core = np.einsum('hkv,hk->hv', expected_state, q, optimize=True)

        actual_state = initial.copy()
        actual_core = delta_net_step(actual_state, q, k, v, decay, beta)
        np.testing.assert_allclose(actual_state, expected_state, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(actual_core, expected_core, rtol=2e-5, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
