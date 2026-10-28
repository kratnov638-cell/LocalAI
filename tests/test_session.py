import struct
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from engine.session import InferenceSession


def gguf_string(value):
    raw = value.encode('utf-8')
    return struct.pack('<Q', len(raw)) + raw


def metadata_entry(key, kind, value):
    out = bytearray(gguf_string(key) + struct.pack('<I', kind))
    if kind == 8:
        out += gguf_string(value)
    elif kind == 4:
        out += struct.pack('<I', value)
    elif kind == 6:
        out += struct.pack('<f', value)
    elif kind == 7:
        out += struct.pack('<?', value)
    elif kind == 9:
        out += struct.pack('<IQ', 8, len(value))
        for item in value:
            out += gguf_string(item)
    else:
        raise AssertionError(kind)
    return out


def write_tiny_gguf(path, architecture="llama"):
    tokens = [chr(i) for i in range(512)] + ['<|im_start|>', '<|im_end|>']
    metadata = [
        ('general.alignment', 4, 32), ('general.architecture', 8, architecture),
        ('llama.block_count', 4, 1), ('llama.attention.head_count', 4, 2),
        ('llama.attention.head_count_kv', 4, 1), ('llama.embedding_length', 4, 4),
        ('llama.feed_forward_length', 4, 8), ('llama.attention.layer_norm_rms_epsilon', 6, 1e-5),
        ('llama.rope.dimension_count', 4, 2), ('llama.rope.freq_base', 6, 10000.0),
        ('tokenizer.ggml.pre', 8, 'qwen2'), ('tokenizer.ggml.tokens', 9, tokens),
        ('tokenizer.ggml.merges', 9, []), ('tokenizer.ggml.bos_token_id', 4, 1),
        ('tokenizer.ggml.eos_token_id', 4, 2), ('tokenizer.ggml.add_bos_token', 7, False),
    ]
    rng = np.random.default_rng(4)
    tensors = [
        ('token_embd.weight', rng.normal(0, .1, (514, 4)).astype('<f4')),
        ('blk.0.attn_norm.weight', np.ones(4, dtype='<f4')),
        ('blk.0.attn_q.weight', rng.normal(0, .1, (4, 4)).astype('<f4')),
        ('blk.0.attn_k.weight', rng.normal(0, .1, (2, 4)).astype('<f4')),
        ('blk.0.attn_v.weight', rng.normal(0, .1, (2, 4)).astype('<f4')),
        ('blk.0.attn_output.weight', rng.normal(0, .1, (4, 4)).astype('<f4')),
        ('blk.0.ffn_norm.weight', np.ones(4, dtype='<f4')),
        ('blk.0.ffn_gate.weight', rng.normal(0, .1, (8, 4)).astype('<f4')),
        ('blk.0.ffn_up.weight', rng.normal(0, .1, (8, 4)).astype('<f4')),
        ('blk.0.ffn_down.weight', rng.normal(0, .1, (4, 8)).astype('<f4')),
        ('output_norm.weight', np.ones(4, dtype='<f4')),
        ('output.weight', np.zeros((514, 4), dtype='<f4')),
        ('output.bias', np.eye(1, 514, 65, dtype='<f4').reshape(514) * 20),
    ]
    info = bytearray()
    data = bytearray()
    offsets = []
    for name, array in tensors:
        data.extend(bytes((-len(data)) % 32))
        offsets.append(len(data))
        data.extend(array.tobytes(order='C'))
    for (name, array), offset in zip(tensors, offsets):
        dims = tuple(reversed(array.shape))
        info += gguf_string(name) + struct.pack('<I', len(dims))
        for dim in dims:
            info += struct.pack('<Q', dim)
        info += struct.pack('<IQ', 0, offset)  # F32, offset relative to data section
    header = bytearray(b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata)))
    for key, kind, value in metadata:
        header += metadata_entry(key, kind, value)
    header += info
    header.extend(bytes((-len(header)) % 32))
    path.write_bytes(header + data)


class SessionIntegrationTests(unittest.TestCase):
    def test_bounded_matvec_and_row_match_full_tensor_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tiny.gguf'
            write_tiny_gguf(path)
            from engine.gguf_parser import GGUFFile
            with GGUFFile(path) as gguf:
                name = 'blk.0.ffn_gate.weight'
                full = gguf.tensor(name)
                x = np.arange(4, dtype=np.float32) / 4
                np.testing.assert_allclose(gguf.matvec(name, x, row_chunk=3), full @ x, rtol=1e-6, atol=1e-6)
                x2 = np.arange(4, dtype=np.float32)[::-1]
                np.testing.assert_allclose(gguf.matvec(name, x2, row_chunk=3), full @ x2, rtol=1e-6, atol=1e-6)
                batched = gguf.matmul(name, np.stack((x, x2)), row_chunk=3)
                np.testing.assert_allclose(batched, np.stack((full @ x, full @ x2)), rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(gguf.tensor_row(name, 2), full[2], rtol=1e-6, atol=1e-6)

    def test_rejects_malformed_qwen35_metadata_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'qwen35.gguf'
            write_tiny_gguf(path, architecture='qwen35')
            with self.assertRaisesRegex(ValueError, 'Missing required Qwen3.5 metadata'):
                InferenceSession(path)

    def test_full_custom_stack_and_weight_cache_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tiny.gguf'
            write_tiny_gguf(path)
            session = InferenceSession(path)
            try:
                prompt_ids = session.tokenizer.encode('Hi', add_bos=False)
                pieces = []
                session.model.generate(prompt_ids, session.tokenizer, threading.Event(), pieces.append,
                                       temperature=0, max_new=3)
                self.assertEqual(''.join(pieces), 'AAA')
                cached = session.model.w('token_embd.weight')
                self.assertIs(session.model.w('token_embd.weight'), cached)
            finally:
                session.close()
            self.assertTrue(session.closed)


if __name__ == '__main__':
    unittest.main()
