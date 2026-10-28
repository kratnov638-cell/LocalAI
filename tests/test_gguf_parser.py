import struct
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from engine.gguf_parser import GGUFFile
from engine.gguf_parser import _native_thread_count


def gguf_string(value):
    raw = value.encode('utf-8')
    return struct.pack('<Q', len(raw)) + raw


class GGUFParserTests(unittest.TestCase):
    def test_decode_thread_override_is_opt_in_and_does_not_change_prefill(self):
        with patch.dict('os.environ', {'LOCALAI_NATIVE_DECODE_THREADS': '8'}):
            self.assertEqual(_native_thread_count(4096, 14), min(8, __import__('os').cpu_count() or 1))
            self.assertEqual(_native_thread_count(4096, 14, batched=True), 6)

    def test_invalid_decode_thread_override_falls_back_to_policy(self):
        with patch.dict('os.environ', {'LOCALAI_NATIVE_DECODE_THREADS': 'not-a-number'}):
            self.assertEqual(_native_thread_count(4096, 14), 7)

    def test_native_thread_policy_reserves_eight_threads_for_large_k_quant_heads(self):
        self.assertEqual(_native_thread_count(128256, 12), 8)
        self.assertEqual(_native_thread_count(128256, 14), 8)

    def test_native_thread_policy_tunes_medium_k_quant_matrices_by_type(self):
        self.assertEqual(_native_thread_count(14336, 12), 8)
        self.assertEqual(_native_thread_count(4096, 12), 8)
        self.assertEqual(_native_thread_count(4096, 14), 7)
        self.assertEqual(_native_thread_count(2048, 14), 6)
        self.assertEqual(_native_thread_count(128256, 8), 6)

    def test_native_batched_thread_policy_avoids_decode_only_thread_overhead(self):
        self.assertEqual(_native_thread_count(14336, 12, batched=True), 6)
        self.assertEqual(_native_thread_count(4096, 14, batched=True), 6)
        self.assertEqual(_native_thread_count(128256, 14, batched=True), 8)

    def test_suffix_lookup_does_not_confuse_attention_output_with_output_head(self):
        gguf = GGUFFile.__new__(GGUFFile)
        gguf.tensors = {'blk.0.attn_output.weight': object()}
        self.assertIsNone(gguf.find('output.weight'))

    def test_v3_metadata_tensor_offsets_and_dequant(self):
        # One Q8_0 block, aligned exactly as a normal GGUF data section.
        header = bytearray(b'GGUF' + struct.pack('<IQQ', 3, 1, 1))
        header += gguf_string('general.alignment') + struct.pack('<I', 4) + struct.pack('<I', 32)
        header += gguf_string('test.weight') + struct.pack('<I', 1)
        header += struct.pack('<QIQ', 32, 8, 0)  # dimension, Q8_0 type, relative offset
        header += bytes((-len(header)) % 32)
        expected = np.arange(-16, 16, dtype=np.int8)
        data = np.float16(.25).tobytes() + expected.tobytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.gguf'
            path.write_bytes(header + data)
            with GGUFFile(path) as gguf:
                self.assertEqual(gguf.version, 3)
                self.assertEqual(gguf.metadata['general.alignment'], 32)
                self.assertEqual(gguf.tensors['test.weight'].shape, (32,))
                np.testing.assert_allclose(gguf.tensor('test.weight'), expected.astype(np.float32) * .25)

    def test_output_head_uses_float32_cache_and_matches_reference(self):
        header = bytearray(b'GGUF' + struct.pack('<IQQ', 3, 1, 1))
        header += gguf_string('general.alignment') + struct.pack('<I', 4) + struct.pack('<I', 32)
        header += gguf_string('output.weight') + struct.pack('<I', 2)
        header += struct.pack('<QQIQ', 32, 32, 1, 0)  # [out, in] F16 matrix
        header += bytes((-len(header)) % 32)
        expected = np.arange(32 * 32, dtype=np.float16).reshape(32, 32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'head.gguf'
            path.write_bytes(header + expected.tobytes())
            with GGUFFile(path) as gguf:
                vector = np.linspace(-1, 1, 32, dtype=np.float32)
                actual = gguf.matvec('output.weight', vector, row_chunk=8)
                np.testing.assert_allclose(actual, expected.astype(np.float32) @ vector)
                self.assertEqual(gguf._head_cache['output.weight'].dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
