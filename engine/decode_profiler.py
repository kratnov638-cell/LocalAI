"""Low-overhead, opt-in timings for autoregressive decode.

Enable with LOCALAI_DECODE_PROFILE=1. The profiler does not alter model math;
it only accumulates wall-clock timings and is inert by default.
"""
from __future__ import annotations

import os
import time


class DecodeProfiler:
    def __init__(self):
        self.enabled = os.environ.get("LOCALAI_DECODE_PROFILE", "").strip().lower() in {"1", "true", "yes", "on"}
        self.reset()

    def reset(self):
        self.active = False
        self.forward_s = 0.0
        self.sampling_s = 0.0
        self.mat_s = {"attention": 0.0, "gate": 0.0, "ffn": 0.0, "ssm": 0.0, "output_head": 0.0, "other": 0.0}
        self.mat_calls = {key: 0 for key in self.mat_s}
        self.tensor_s = {}
        self.tensor_calls = {}
        self.forward_calls = 0

    def start_decode(self):
        if self.enabled:
            self.reset()
            self.active = True

    def record_forward(self, elapsed):
        if self.enabled and self.active:
            self.forward_s += elapsed
            self.forward_calls += 1

    def record_sampling(self, elapsed):
        if self.enabled and self.active:
            self.sampling_s += elapsed

    @staticmethod
    def category(name):
        lower = name.lower()
        if lower == "output.weight" or lower.endswith(".output.weight"):
            return "output_head"
        if "attn_gate" in lower:
            return "gate"
        if any(part in lower for part in ("ssm_", "attn_qkv", "ssm.")):
            return "ssm"
        if any(part in lower for part in ("attn_", "attention", "attn.")):
            return "attention"
        if any(part in lower for part in ("ffn_", "feed_forward", "mlp.")):
            return "ffn"
        return "other"

    def record_mat(self, name, elapsed):
        if self.enabled and self.active:
            key = self.category(name)
            self.mat_s[key] += elapsed
            self.mat_calls[key] += 1
            self.tensor_s[name] = self.tensor_s.get(name, 0.0) + elapsed
            self.tensor_calls[name] = self.tensor_calls.get(name, 0) + 1

    def report(self):
        if not self.enabled:
            return None
        mat_total = sum(self.mat_s.values())
        non_mat = max(0.0, self.forward_s - mat_total)
        return {
            "forward_s": self.forward_s,
            "forward_calls": self.forward_calls,
            "sampling_s": self.sampling_s,
            "mat_s": dict(self.mat_s),
            "mat_calls": dict(self.mat_calls),
            "top_matrices": [
                {"name": name, "seconds": elapsed, "calls": self.tensor_calls[name]}
                for name, elapsed in sorted(self.tensor_s.items(), key=lambda item: item[1], reverse=True)[:12]
            ],
            "non_mat_s": non_mat,
        }
