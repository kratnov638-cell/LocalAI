import os
import unittest
from unittest.mock import patch

from engine.decode_profiler import DecodeProfiler


class DecodeProfilerTests(unittest.TestCase):
    def test_disabled_by_default_and_report_is_absent(self):
        with patch.dict(os.environ, {}, clear=True):
            profiler = DecodeProfiler()
        self.assertFalse(profiler.enabled)
        self.assertIsNone(profiler.report())

    def test_aggregates_only_while_decode_is_active(self):
        with patch.dict(os.environ, {"LOCALAI_DECODE_PROFILE": "1"}):
            profiler = DecodeProfiler()
        profiler.record_mat("blk.0.ffn_gate.weight", 1.0)
        profiler.start_decode()
        profiler.record_mat("blk.0.ffn_gate.weight", 0.25)
        profiler.record_mat("blk.0.attn_qkv.weight", 0.5)
        profiler.record_mat("output.weight", 0.75)
        profiler.record_forward(2.0)
        profiler.record_sampling(0.1)
        report = profiler.report()
        self.assertEqual(report["mat_calls"]["ffn"], 1)
        self.assertEqual(report["mat_calls"]["ssm"], 1)
        self.assertEqual(report["mat_calls"]["output_head"], 1)
        self.assertAlmostEqual(report["mat_s"]["ffn"], 0.25)
        self.assertEqual(report["top_matrices"][0]["name"], "output.weight")
        self.assertEqual(report["top_matrices"][0]["calls"], 1)
        self.assertAlmostEqual(report["forward_s"], 2.0)
        self.assertAlmostEqual(report["sampling_s"], 0.1)
        self.assertAlmostEqual(report["non_mat_s"], 0.5)


if __name__ == "__main__":
    unittest.main()
