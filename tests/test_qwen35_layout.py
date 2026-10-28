import unittest

from engine.qwen35_layout import Qwen35Layout


class Qwen35LayoutTests(unittest.TestCase):
    def test_inspects_hybrid_layer_pattern_and_vision_presence(self):
        metadata = {
            'general.architecture': 'qwen35',
            'qwen35.embedding_length': 8,
            'qwen35.block_count': 4,
            'qwen35.full_attention_interval': 4,
            'tokenizer.ggml.tokens': ['a', 'b'],
        }
        tensors = {}
        for i in range(4):
            prefix = f'blk.{i}.'
            tensors[prefix + 'attn_norm.weight'] = object()
            if i == 3:
                for suffix in ('attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight'):
                    tensors[prefix + suffix] = object()
            else:
                for suffix in ('attn_qkv.weight', 'ssm_a', 'ssm_conv1d.weight', 'ssm_out.weight'):
                    tensors[prefix + suffix] = object()
        tensors['v.patch_embd.weight'] = object()
        layout = Qwen35Layout.inspect(metadata, tensors)
        self.assertEqual(layout.full_attention_layers, (3,))
        self.assertEqual(layout.linear_attention_layers, (0, 1, 2))
        self.assertTrue(layout.has_vision_tower)
        self.assertEqual(layout.hidden_size, 8)

    def test_rejects_layer_tensor_pattern_conflicting_with_metadata(self):
        metadata = {
            'general.architecture': 'qwen35',
            'qwen35.embedding_length': 8,
            'qwen35.block_count': 4,
            'qwen35.full_attention_interval': 4,
            'tokenizer.ggml.tokens': ['a'],
        }
        tensors = {'blk.0.attn_q.weight': object()}
        with self.assertRaisesRegex(ValueError, 'ambiguous/incomplete|conflicts'):
            Qwen35Layout.inspect(metadata, tensors)

    def test_rejects_wrong_architecture(self):
        with self.assertRaisesRegex(ValueError, 'requires qwen35'):
            Qwen35Layout.inspect({'general.architecture': 'llama'}, {})
