"""CPU inference for the legacy 80-entry ZAYA GGUF layout.

The GGUF stores alternating CCA and sparse-MoE entries: even indices contain
CCA tensors and odd indices contain router/expert tensors. This is deliberately
separate from LlamaModel; sharing the sampler is fine, sharing its forward pass
is not.
"""
from __future__ import annotations

import math
from collections import OrderedDict
import numpy as np

from .model import LlamaModel, KVCache
from .dequant import block_layout, dequantize


class ZayaModel(LlamaModel):
    def __init__(self, gguf, max_seq=2048):
        super().__init__(gguf, max_seq=max_seq)
        m = gguf.metadata
        if self.arch != 'zaya' or self.n_layers != 80 or self.dim != 2048:
            raise ValueError('ZayaModel currently supports the legacy 80-entry ZAYA1-8B GGUF layout only')
        if self.n_heads != 8 or self.n_kv_heads != 2 or int(m.get('zaya.expert_count', 0)) != 16:
            raise ValueError('Unsupported ZAYA dimensions or expert count')
        self.head_dim = int(m.get('zaya.attention.key_length', 128))
        self.rope_dim = int(m.get('zaya.rope.dimension_count', 64))
        self.rope_base = float(m.get('zaya.rope.freq_base', 5_000_000.0))
        self.eps = float(m.get('zaya.attention.layer_norm_rms_epsilon', 1e-5))
        self.n_experts = int(m['zaya.expert_count'])
        self.expert_dim = int(m.get('zaya.expert_feed_forward_length', 256))
        self.router_dim = 256
        self.cca_layers = tuple(range(0, self.n_layers, 2))
        self.moe_layers = tuple(range(1, self.n_layers, 2))
        self._conv_state = np.zeros((self.n_layers, 2, self.n_heads * self.head_dim + self.n_kv_heads * self.head_dim), np.float32)
        # LlamaModel derives head_dim as hidden_size / query_heads (256 here),
        # while legacy ZAYA uses an explicit 128-wide head and 64 rotary dims.
        self.cache = KVCache(self.n_layers, self.n_kv_heads, self.head_dim, max_seq)
        self._prev_hidden = np.zeros((self.n_layers, self.dim), np.float32)
        # Keep a small LRU of selected expert matrices. The native Q6_K kernel
        # below normally avoids dequantization entirely; this cache is only for
        # the portable fallback and is bounded because each expert is large.
        self._expert_cache = OrderedDict()
        self._expert_cache_limit = 4
        self._rope_inv = 1.0 / (self.rope_base ** (np.arange(self.rope_dim // 2, dtype=np.float32) / (self.rope_dim // 2)))
        # Fail early on incomplete/mismatched tensor mapping instead of generating garbage.
        required = ['token_embd.weight', 'output_norm.weight', 'res_scale_hs.weight',
                    'res_scale_hs.bias', 'res_scale_res.weight', 'res_scale_res.bias']
        required += [f'blk.{i}.{name}' for i in self.cca_layers for name in (
            'attn_norm.weight', 'attn_q.weight', 'attn_k.weight', 'attn_output.weight',
            'cca_val_proj1.weight', 'cca_val_proj2.weight', 'cca_conv_grp.weight',
            'cca_conv_grp.bias', 'cca_k_scale.weight', 'ssm_conv1d.weight', 'ssm_conv1d.bias',
            'res_scale_hs.weight', 'res_scale_hs.bias')]
        required += [f'blk.{i}.{name}' for i in self.moe_layers for name in (
            'attn_norm.weight', 'ffn_norm.weight', 'ffn_gate_inp.weight', 'ffn_gate_inp.bias',
            'ffn_gate.weight', 'ffn_gate.bias', 'zaya_router_mlp2.weight', 'zaya_router_mlp2.bias',
            'zaya_router_mlp4.weight', 'ffn_gate_up_exps.weight', 'ffn_down_exps.weight',
            'zaya_router_biases.weight', 'res_scale_hs.weight', 'res_scale_hs.bias',
            'res_scale_res.weight', 'res_scale_res.bias')]
        missing = [name for name in required if name not in gguf.tensors]
        if missing:
            raise ValueError(f'ZAYA GGUF tensor mapping incomplete ({len(missing)} missing): {missing[:12]}')

    def reset_state(self):
        self.cache.reset()
        self._conv_state.fill(0)
        self._prev_hidden.fill(0)

    def _expert_matvec(self, name, expert, x):
        """Dequantize only one packed expert, not the entire 16-expert tensor."""
        info = self.gguf.tensors[name]
        if len(info.shape) != 3 or info.shape[0] != self.n_experts:
            raise ValueError(f'{name}: expected [experts,out,in], got {info.shape}')
        _, bpb = block_layout(info.ggml_type)
        elements_per_expert = int(np.prod(info.shape[1:], dtype=np.int64))
        if elements_per_expert % block_layout(info.ggml_type)[0]:
            raise ValueError(f'{name}: expert dimensions do not align to quant blocks')
        bytes_per_expert = (elements_per_expert // block_layout(info.ggml_type)[0]) * bpb
        start = info.offset + int(expert) * bytes_per_expert
        raw = memoryview(self.gguf._map)[start:start + bytes_per_expert]
        x = np.asarray(x, dtype=np.float32)
        rows, cols = info.shape[1:]
        if x.size != cols:
            raise ValueError(f'{name}: expert input has {x.size} values, expected {cols}')
        # This is the hot path: the selected expert is a contiguous quantized
        # matrix slice, so it can use the same project-owned AVX2 Q6_K kernel
        # as ordinary 2-D GGUF matrices without expanding weights to float32.
        if info.ggml_type in (2, 8, 12, 13, 14):
            try:
                from .native_quant import matvec as native_matvec
                result = native_matvec(raw, x, rows, cols, info.ggml_type)
                if result is not None:
                    return result
            except Exception:
                pass
        key = (name, int(expert))
        w = self._expert_cache.get(key)
        if w is None:
            w = dequantize(raw, info.ggml_type, info.shape[1:])
            self._expert_cache[key] = w
            if len(self._expert_cache) > self._expert_cache_limit:
                self._expert_cache.popitem(last=False)
        else:
            self._expert_cache.move_to_end(key)
        return w @ x

    def _matvec(self, name, x):
        return self.mat(np.asarray(x, dtype=np.float32), name)

    @staticmethod
    def _gelu(x):
        # Exact GELU matches torch.nn.GELU's default (not the tanh approximation).
        from math import erf
        flat = np.asarray(x, dtype=np.float32).reshape(-1)
        return (np.fromiter((v * (1.0 + erf(float(v) / math.sqrt(2.0))) * 0.5 for v in flat),
                            dtype=np.float32, count=flat.size)).reshape(np.shape(x))

    @staticmethod
    def _res_scale(branch, scale, bias):
        return (branch + bias) * scale

    def _cca(self, layer, h, pos):
        p = f'blk.{layer}.'
        qraw = self._matvec(p + 'attn_q.weight', h)
        kraw = self._matvec(p + 'attn_k.weight', h)
        raw = np.concatenate((qraw, kraw)).astype(np.float32, copy=False)
        state = self._conv_state[layer]
        # Depthwise causal 2-tap convolution is evaluated for previous and current
        # positions, then the grouped 2-tap convolution mixes channels per head.
        # Left padding in the reference Conv1d stack is literal zeros. It is
        # not the output of the first convolution at a fictitious previous
        # position (which would incorrectly inject the depthwise bias).
        if pos == 0:
            prev_depth = np.zeros_like(raw)
        else:
            prev_depth = (state[0] * self.w(p + 'ssm_conv1d.weight')[:, 0]
                          + state[1] * self.w(p + 'ssm_conv1d.weight')[:, 1]
                          + self.w(p + 'ssm_conv1d.bias'))
        cur_depth = state[1] * self.w(p + 'ssm_conv1d.weight')[:, 0] + raw * self.w(p + 'ssm_conv1d.weight')[:, 1] + self.w(p + 'ssm_conv1d.bias')
        self._conv_state[layer, 0] = state[1]
        self._conv_state[layer, 1] = raw
        conv_in = np.stack((prev_depth, cur_depth), axis=1)
        grouped_w = self.w(p + 'cca_conv_grp.weight')
        grouped = np.empty(1280, dtype=np.float32)
        # 8 query heads + 2 KV heads = 10 groups, 128 channels per group.
        for group in range(10):
            lo = group * 128
            grouped[lo:lo+128] = np.einsum('ock,ck->o', grouped_w[lo:lo+128], conv_in[lo:lo+128], optimize=True)
        grouped += self.w(p + 'cca_conv_grp.bias')
        q_conv, k_conv = grouped[:1024], grouped[1024:]
        q_pre = qraw.reshape(8, 128)
        k_pre = kraw.reshape(2, 128)
        k_rep = np.repeat(k_pre, 4, axis=0)
        q_res = (q_pre + k_rep) * 0.5
        k_res = q_res.reshape(2, 4, 128).mean(axis=1)
        q = q_conv.reshape(8, 128) + q_res
        k = k_conv.reshape(2, 128) + k_res
        qnorm = np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
        knorm = np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1e-12)
        q = q * (math.sqrt(128.0) / qnorm)
        k = k * (math.sqrt(128.0) / knorm) * self.w(p + 'cca_k_scale.weight')[:, None]
        rotary = self.rope_dim // 2
        angle = np.float32(pos) * self._rope_inv
        c, s = np.cos(angle), np.sin(angle)
        def rope(a):
            out = a.copy(); left = a[..., :rotary].copy(); right = a[..., rotary:2*rotary].copy()
            out[..., :rotary] = left * c - right * s
            out[..., rotary:2*rotary] = left * s + right * c
            return out
        q, k = rope(q), rope(k)
        self.cache.k[layer, pos] = k
        # CCA value path is half current projection and half delayed projection
        # of the previous layer input, matching the recurrent CCA state.
        vcur = self._matvec(p + 'cca_val_proj1.weight', h)
        vdel = self._matvec(p + 'cca_val_proj2.weight', self._prev_hidden[layer])
        self._prev_hidden[layer] = h
        value = np.concatenate((vcur, vdel)).reshape(2, 128)
        self.cache.v[layer, pos] = value
        ks = np.repeat(self.cache.k[layer, :pos+1], 4, axis=1).transpose(1, 0, 2)
        vs = np.repeat(self.cache.v[layer, :pos+1], 4, axis=1).transpose(1, 0, 2)
        scores = np.einsum('hd,hld->hl', q, ks, optimize=True) / math.sqrt(128.0)
        scores -= scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores); probs /= probs.sum(axis=-1, keepdims=True)
        att = np.einsum('hl,hld->hd', probs, vs, optimize=True).reshape(1024)
        out = self._matvec(p + 'attn_output.weight', att)
        return out

    def _moe(self, layer, h, prev_router):
        p = f'blk.{layer}.'
        # LlamaModel.mat() already adds a matching GGUF bias tensor. Do not
        # add router biases a second time here: double biasing every router
        # projection changes the expert distribution at all 40 MoE layers.
        router = self._matvec(p + 'ffn_gate_inp.weight', h)
        eda_name = p + 'zaya_router_eda.weight'
        # EDA passes the *current token's* pre-norm router state from the
        # preceding MoE layer, not a state from the preceding time step.
        if eda_name in self.gguf.tensors and prev_router is not None:
            router = router + prev_router * self.w(eda_name)
        router_state = router.copy()
        router = self.norm(router, self.w(p + 'ffn_norm.weight'), self.eps)
        router = self._gelu(self._matvec(p + 'ffn_gate.weight', router))
        router = self._gelu(self._matvec(p + 'zaya_router_mlp2.weight', router))
        logits = self._matvec(p + 'zaya_router_mlp4.weight', router)
        logits -= logits.max(); probs = np.exp(logits); probs /= probs.sum()
        # The GGUF stores the learned per-expert routing biases, including the
        # -1 skip-expert bias, separately from the MLP logits. Apply them to
        # probabilities for selection; weight the selected expert by the
        # original (unbiased) probability, as the reference implementation does.
        route = probs + self.w(p + 'zaya_router_biases.weight')
        expert = int(np.argmax(route))
        if expert == self.n_experts:
            # MOD's final expert is a learned skip route. The reference model
            # forwards the MoE block input through this route (then applies the
            # selected expert probability), rather than returning zero.
            moe = h.copy() * probs[expert]
        else:
            packed = self._expert_matvec(p + 'ffn_gate_up_exps.weight', expert, h)
            gate, up = np.split(packed, 2)
            silu = gate / (1.0 + np.exp(-np.clip(gate, -80, 80)))
            moe = self._expert_matvec(p + 'ffn_down_exps.weight', expert, silu * up) * probs[expert]
        return moe, router_state

    def forward(self, token, position):
        emb = self.gguf.find('token_embd.weight', 'tok_embeddings.weight')
        hidden = self.gguf.tensor_row(emb, int(token)).astype(np.float32, copy=False)
        residual = None
        prev_router = None
        for layer in range(self.n_layers):
            p = f'blk.{layer}.'
            # Match the legacy Transformers residual-stream contract: each
            # layer scales its branch and (except layer 0) the carried residual,
            # merges them, then normalizes before attention/router computation.
            hidden = self._res_scale(hidden, self.w(p + 'res_scale_hs.weight'), self.w(p + 'res_scale_hs.bias'))
            if layer > 0:
                residual = self._res_scale(residual, self.w(p + 'res_scale_res.weight'), self.w(p + 'res_scale_res.bias'))
            if residual is None:
                residual = hidden.copy()
            else:
                residual = residual + hidden
            h = self.norm(residual, self.w(p + 'attn_norm.weight'), self.eps)
            if layer % 2 == 0:
                hidden = self._cca(layer, h, position)
            else:
                hidden, prev_router = self._moe(layer, h, prev_router)
        residual = self._res_scale(residual, self.w('res_scale_res.weight'), self.w('res_scale_res.bias'))
        hidden = self._res_scale(hidden, self.w('res_scale_hs.weight'), self.w('res_scale_hs.bias'))
        out = self.norm(residual + hidden, self.w('output_norm.weight'), self.eps)
        return self.gguf.matvec(emb, out)

    def generate(self, *args, **kwargs):
        self.reset_state()
        return super().generate(*args, **kwargs)
