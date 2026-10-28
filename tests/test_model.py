import unittest
import numpy as np

from engine.model import LlamaModel, Qwen2Model, Qwen3Model


class FakeGGUF:
    def __init__(self):
        self.metadata = {
            'general.architecture': 'llama',
            'llama.block_count': 1,
            'llama.attention.head_count': 2,
            'llama.attention.head_count_kv': 1,
            'llama.embedding_length': 4,
            'llama.feed_forward_length': 8,
            'llama.attention.layer_norm_rms_epsilon': 1e-5,
            'llama.rope.dimension_count': 2,
            'tokenizer.ggml.vocab_size': 16,
        }
        rng = np.random.default_rng(7)
        def matrix(name, shape): self.tensors[name] = rng.normal(0, .1, shape).astype(np.float32)
        self.tensors = {}
        matrix('token_embd.weight', (16, 4))
        matrix('blk.0.attn_norm.weight', (4,))
        matrix('blk.0.attn_q.weight', (4, 4))
        matrix('blk.0.attn_k.weight', (2, 4))
        matrix('blk.0.attn_v.weight', (2, 4))
        matrix('blk.0.attn_output.weight', (4, 4))
        matrix('blk.0.ffn_norm.weight', (4,))
        matrix('blk.0.ffn_gate.weight', (8, 4))
        matrix('blk.0.ffn_up.weight', (8, 4))
        matrix('blk.0.ffn_down.weight', (4, 8))
        matrix('output_norm.weight', (4,))
        matrix('output.weight', (16, 4))

    def tensor(self, name): return self.tensors[name]
    def find(self, *suffixes):
        for suffix in suffixes:
            if suffix in self.tensors: return suffix
        for suffix in suffixes:
            for name in self.tensors:
                if name.endswith('.' + suffix): return name
        return None


class ModelTests(unittest.TestCase):
    def test_llama_rope_rotates_adjacent_pairs(self):
        gguf = FakeGGUF()
        gguf.metadata['llama.rope.dimension_count'] = 4
        gguf.metadata['llama.attention.head_count'] = 1
        model = LlamaModel(gguf, max_seq=8)
        x = np.array([[[1., 2., 3., 4.]]], dtype=np.float32)
        got = model._rope_sequence(x, np.array([1]))
        inv = 1.0 / (model.rope_base ** (np.arange(2, dtype=np.float32) / 2))
        c, s = np.cos(inv), np.sin(inv)
        expected = np.array([[[1*c[0]-2*s[0], 1*s[0]+2*c[0], 3*c[1]-4*s[1], 3*s[1]+4*c[1]]]], dtype=np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_qwen_rope_keeps_split_half_layout(self):
        gguf = FakeGGUF()
        gguf.metadata['general.architecture'] = 'qwen2'
        gguf.metadata.pop('llama.rope.freq_base', None)
        gguf.metadata['llama.rope.dimension_count'] = 4
        gguf.metadata['llama.attention.head_count'] = 1
        model = Qwen2Model(gguf, max_seq=8)
        x = np.array([[[1., 2., 3., 4.]]], dtype=np.float32)
        got = model._rope_sequence(x, np.array([1]))
        inv = 1.0 / (model.rope_base ** (np.arange(2, dtype=np.float32) / 2))
        c, s = np.cos(inv), np.sin(inv)
        expected = np.array([[[1*c[0]-3*s[0], 2*c[1]-4*s[1], 1*s[0]+3*c[0], 2*s[1]+4*c[1]]]], dtype=np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_optional_qwen_projection_bias(self):
        gguf = FakeGGUF()
        bias = np.array([1., 2., 3., 4.], dtype=np.float32)
        gguf.tensors['blk.0.attn_q.bias'] = bias
        model = LlamaModel(gguf)
        np.testing.assert_allclose(model.mat(np.zeros(4, dtype=np.float32), 'blk.0.attn_q.weight'), bias)

    def test_forward_shape_and_kv_cache(self):
        model = LlamaModel(FakeGGUF(), max_seq=8)
        logits = model.forward(3, 0)
        self.assertEqual(logits.shape, (16,))
        self.assertTrue(np.isfinite(logits).all())
        self.assertTrue(np.any(model.cache.k[0, 0] != 0))
        model.cache.length = 1
        logits2 = model.forward(4, 1)
        self.assertEqual(logits2.shape, (16,))
        self.assertTrue(np.isfinite(logits2).all())


    def test_batched_prefill_matches_tokenwise_logits_and_cache(self):
        tokens = [3, 7, 4, 2]
        batched = LlamaModel(FakeGGUF(), max_seq=8)
        batched_logits = batched.forward_sequence(tokens)
        tokenwise = LlamaModel(FakeGGUF(), max_seq=8)
        tokenwise_logits = None
        for position, token in enumerate(tokens):
            tokenwise_logits = tokenwise.forward(token, position)
        np.testing.assert_allclose(batched_logits, tokenwise_logits, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(batched.cache.k[:, :len(tokens)], tokenwise.cache.k[:, :len(tokens)], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(batched.cache.v[:, :len(tokens)], tokenwise.cache.v[:, :len(tokens)], rtol=2e-5, atol=2e-6)

    def test_qwen2_uses_its_own_class_and_default_rope_theta(self):
        gguf = FakeGGUF()
        gguf.metadata['general.architecture'] = 'qwen2'
        gguf.metadata.pop('llama.rope.freq_base', None)
        model = Qwen2Model(gguf, max_seq=8)
        self.assertEqual(model.rope_base, 1_000_000.0)

    def test_qwen2_class_rejects_other_architectures(self):
        with self.assertRaisesRegex(ValueError, 'requires general.architecture=qwen2'):
            Qwen2Model(FakeGGUF())

    def test_qwen3_uses_qk_norms_and_its_rope_default(self):
        gguf = FakeGGUF()
        gguf.metadata['general.architecture'] = 'qwen3'
        gguf.metadata.pop('llama.rope.freq_base', None)
        gguf.tensors['blk.0.post_attention_norm.weight'] = gguf.tensors.pop('blk.0.ffn_norm.weight')
        gguf.tensors['blk.0.attn_q_norm.weight'] = np.ones(2, dtype=np.float32)
        gguf.tensors['blk.0.attn_k_norm.weight'] = np.ones(2, dtype=np.float32)
        model = Qwen3Model(gguf, max_seq=8)
        self.assertEqual(model.rope_base, 1_000_000.0)
        logits = model.forward(3, 0)
        self.assertEqual(logits.shape, (16,))
        self.assertTrue(np.isfinite(logits).all())

    def test_llama3_rope_scaling_path(self):
        gguf = FakeGGUF()
        gguf.metadata.update({
            'llama.rope.scaling.type': 'llama3',
            'llama.rope.scaling.factor': 32.0,
            'llama.rope.scaling.original_context_length': 8192,
            'llama.rope.scaling.low_freq_factor': 1.0,
            'llama.rope.scaling.high_freq_factor': 4.0,
        })
        model = LlamaModel(gguf, max_seq=8)
        logits = model.forward(3, 7)
        self.assertEqual(logits.shape, (16,))
        self.assertTrue(np.isfinite(logits).all())


if __name__ == '__main__':
    unittest.main()
