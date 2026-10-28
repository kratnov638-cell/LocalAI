"""NumPy-only text decoder for dense Qwen3.5 GGUF checkpoints.

Vision inputs are intentionally not implemented here. Weight matrices are
multiplied in bounded row chunks by GGUFFile.matvec to avoid expanding the
whole quantized checkpoint into RAM.
"""
from __future__ import annotations

import math
import time
import numpy as np

from .qwen35_layout import Qwen35Layout
from .decode_profiler import DecodeProfiler


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def _silu(x):
    return x * _sigmoid(x)


def _softplus(x):
    return np.logaddexp(np.asarray(x, dtype=np.float32), 0.0)


def _rms(x, weight, eps, *, unit_offset=False):
    scale = weight + 1.0 if unit_offset else weight
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * scale


class Qwen35Model:
    """Dense Qwen3.5 text generation; multimodal vision is a separate feature."""

    def __init__(self, gguf, max_seq=2048):
        self.gguf = gguf
        self.arch = str(gguf.metadata.get('general.architecture', '')).lower()
        if self.arch not in {'qwen35', 'qwen3.5'}:
            raise ValueError(f'Qwen35Model requires qwen35 architecture, got {self.arch!r}')
        m = gguf.metadata
        self.metadata_prefix = 'qwen35.' if any(key.startswith('qwen35.') for key in m) else 'qwen3.5.'
        def meta(key, default=None): return m.get(self.metadata_prefix + key, default)
        self.layout = Qwen35Layout.inspect(m, gguf.tensors)
        self.n_layers = int(meta('block_count'))
        self.dim = int(meta('embedding_length'))
        self.ffn = int(meta('feed_forward_length'))
        self.n_heads = int(meta('attention.head_count'))
        self.head_dim = int(meta('attention.key_length', self.dim // self.n_heads))
        kv_meta = meta('attention.head_count_kv', 4)
        self.n_kv_heads = max(int(v) for v in kv_meta) if isinstance(kv_meta, (list, tuple)) else int(kv_meta)
        self.rope_dim = int(meta('rope.dimension_count', 64))
        self.rope_base = float(meta('rope.freq_base', 10_000_000.0))
        self.eps = float(meta('attention.layer_norm_rms_epsilon', 1e-6))
        self.conv_kernel = int(meta('ssm.conv_kernel', 4))
        self.ssm_key_heads = int(meta('ssm.group_count', 16))
        self.ssm_state_size = int(meta('ssm.state_size', 128))
        self.ssm_value_heads = int(meta('ssm.inner_size', 4096)) // self.ssm_state_size
        self.ssm_key_dim = self.ssm_state_size
        self.ssm_value_dim = self.ssm_state_size
        self.max_seq = int(max_seq)
        if not all((self.dim, self.ffn, self.n_heads, self.head_dim, self.n_layers)):
            raise ValueError('Qwen3.5 GGUF is missing required dimensions')
        if self.n_heads * self.head_dim != self.dim:
            raise ValueError(f'Qwen3.5 head dimensions mismatch: {self.n_heads}*{self.head_dim}!={self.dim}')
        if self.ssm_value_heads * self.ssm_value_dim != int(meta('ssm.inner_size', 4096)):
            raise ValueError('This Qwen3.5 GGUF has an unexpected Gated DeltaNet value width')
        linear_name = 'blk.0.attn_qkv.weight'
        if linear_name in gguf.tensors:
            linear_width = gguf.tensors[linear_name].shape[0]
            expected_width = self.ssm_key_heads * self.ssm_key_dim * 2 + self.ssm_value_heads * self.ssm_value_dim
            if linear_width != expected_width:
                raise ValueError(f'{linear_name} output width {linear_width} != configured Gated DeltaNet width {expected_width}')
        else:
            raise ValueError(f'Qwen3.5 GGUF is missing required tensor {linear_name}')
        self._weights = {}
        self._delta_net_step_impl = None
        self._delta_net_step_checked = False
        self.decode_profiler = DecodeProfiler()
        self.progress_callback = None
        self.full_layers = set(self.layout.full_attention_layers)
        self.k_cache = {i: np.zeros((self.max_seq, self.n_kv_heads, self.head_dim), np.float32)
                        for i in self.full_layers}
        self.v_cache = {i: np.zeros_like(v) for i, v in self.k_cache.items()}
        self.conv_state = {i: np.zeros((self.conv_kernel - 1, expected_width), np.float32)
                           for i in self.layout.linear_attention_layers}
        self.recurrent_state = {i: np.zeros((self.ssm_value_heads, self.ssm_key_dim, self.ssm_value_dim), np.float32)
                                for i in self.layout.linear_attention_layers}
        self.cache_length = 0

    def w(self, name):
        if name not in self._weights:
            # GGUF exporters disagree on whether the learned DeltaNet time-step
            # bias is named `ssm_dt` or `ssm_dt.bias`. Resolve this at the model
            # boundary instead of assuming one checkpoint's spelling.
            candidates = [name]
            if name.endswith('.ssm_dt'):
                candidates.append(name + '.bias')
            elif name.endswith('.ssm_dt.bias'):
                candidates.append(name[:-5])
            resolved = next((candidate for candidate in candidates if candidate in self.gguf.tensors), None)
            if resolved is None:
                raise KeyError(f'Missing Qwen3.5 tensor {name!r}; tried {candidates}')
            self._weights[name] = self.gguf.tensor(resolved)
        return self._weights[name]

    def mat(self, x, name):
        profiler = getattr(self, 'decode_profiler', None)
        if profiler is None or not profiler.enabled or not profiler.active:
            return self._mat_impl(x, name)
        started = time.perf_counter()
        try:
            return self._mat_impl(x, name)
        finally:
            profiler.record_mat(name, time.perf_counter() - started)

    def _mat_impl(self, x, name):
        info = self.gguf.tensors[name]
        if len(info.shape) != 2:
            raise ValueError(f'{name}: expected matrix, got {info.shape}')
        if x.ndim == 1 and info.shape[1] == x.size and hasattr(self.gguf, 'matvec'):
            y = self.gguf.matvec(name, x)
        elif x.ndim == 2 and info.shape[1] == x.shape[-1] and hasattr(self.gguf, 'matmul'):
            # Prompt prefill reuses each decoded weight tile across all tokens
            # instead of issuing one memory-bound GEMV per prompt token.
            y = self.gguf.matmul(name, x)
        else:
            weights = self.w(name)
            if weights.shape[1] == x.shape[-1]: y = x @ weights.T
            elif weights.shape[0] == x.shape[-1]: y = x @ weights
            else: raise ValueError(f'{name}: matrix shape {weights.shape} incompatible with {x.shape}')
        bias_name = name[:-6] + 'bias' if name.endswith('.weight') else ''
        if bias_name and bias_name in self.gguf.tensors:
            y = y + self.w(bias_name)
        return y

    def norm(self, x, name, *, unit_offset=False):
        return _rms(x, self.w(name), self.eps, unit_offset=unit_offset)

    def _rope(self, a, position):
        rotary = min(self.rope_dim, self.head_dim)
        rotary -= rotary % 2
        half = rotary // 2
        inv = 1.0 / (self.rope_base ** (np.arange(half, dtype=np.float32) / max(half, 1)))
        angle = position * inv
        c, s = np.cos(angle), np.sin(angle)
        out = a.copy()
        # Text-only MRoPE has identical T/H/W positions and reduces to the
        # standard split-half rotate_half operation used by Qwen3.5.
        left = a[..., :half].copy()
        right = a[..., half:rotary].copy()
        out[..., :half] = left * c - right * s
        out[..., half:rotary] = left * s + right * c
        return out

    def _full_attention(self, x, layer, position):
        p = f'blk.{layer}.'
        qg = self.mat(x, p + 'attn_q.weight').reshape(self.n_heads, self.head_dim * 2)
        q = qg[:, :self.head_dim]
        gate = qg[:, self.head_dim:]
        k = self.mat(x, p + 'attn_k.weight').reshape(self.n_kv_heads, self.head_dim)
        v = self.mat(x, p + 'attn_v.weight').reshape(self.n_kv_heads, self.head_dim)
        q = _rms(q, self.w(p + 'attn_q_norm.weight'), self.eps)
        k = _rms(k, self.w(p + 'attn_k_norm.weight'), self.eps)
        q = self._rope(q, position)
        k = self._rope(k, position)
        self.k_cache[layer][position] = k
        self.v_cache[layer][position] = v
        keys = self.k_cache[layer][:position + 1]
        values = self.v_cache[layer][:position + 1]
        group = self.n_heads // self.n_kv_heads
        # Fold query groups into the head dimension without copying K/V to all heads.
        q_grouped = q.reshape(self.n_kv_heads, group, self.head_dim)
        scores = np.einsum('hgd,thd->hgt', q_grouped, keys, optimize=True) / math.sqrt(self.head_dim)
        scores -= scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=-1, keepdims=True)
        att = np.einsum('hgt,thd->hgd', probs, values, optimize=True).reshape(self.dim)
        att *= _sigmoid(gate.reshape(self.dim))
        return self.mat(att, p + 'attn_output.weight')

    def _linear_attention(self, x, layer):
        p = f'blk.{layer}.'
        qkv = self.mat(x, p + 'attn_qkv.weight')
        z = self.mat(x, p + 'attn_gate.weight')
        beta_logits = self.mat(x, p + 'ssm_beta.weight')
        alpha_logits = self.mat(x, p + 'ssm_alpha.weight')
        key_width = self.ssm_key_heads * self.ssm_key_dim
        value_width = self.ssm_value_heads * self.ssm_value_dim
        if qkv.size != key_width * 2 + value_width:
            raise ValueError(f'{p}attn_qkv.weight output width {qkv.size} is not expected {key_width * 2 + value_width}')
        raw = np.concatenate((self.conv_state[layer], qkv[None, :]), axis=0)
        conv_w = self.w(p + 'ssm_conv1d.weight')
        if conv_w.shape != (qkv.size, self.conv_kernel):
            raise ValueError(f'{p}ssm_conv1d.weight shape {conv_w.shape} != {(qkv.size, self.conv_kernel)}')
        convolved = np.sum(raw.T * conv_w, axis=1)
        self.conv_state[layer] = raw[1:].copy()
        mixed = _silu(convolved)
        q, k, v = np.split(mixed, (key_width, key_width * 2))
        q = q.reshape(self.ssm_key_heads, self.ssm_key_dim)
        k = k.reshape(self.ssm_key_heads, self.ssm_key_dim)
        v = v.reshape(self.ssm_value_heads, self.ssm_value_dim)
        # Q/K are L2-normalized in the recurrent DeltaNet reference path.
        q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-6)
        k /= np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1e-6)
        q /= math.sqrt(self.ssm_key_dim)
        if self.ssm_value_heads % self.ssm_key_heads:
            raise ValueError('Gated DeltaNet value heads must be a multiple of key heads')
        repeats = self.ssm_value_heads // self.ssm_key_heads
        # The stored head order changes which key head each value head pairs
        # with; using the wrong mapping silently corrupts the recurrent state.
        if bool(self.gguf.metadata.get(self.metadata_prefix + 'ssm.v_head_reordered', True)):
            head_map = np.tile(np.arange(self.ssm_key_heads), repeats)
        else:
            head_map = np.repeat(np.arange(self.ssm_key_heads), repeats)
        q = q[head_map]
        k = k[head_map]
        a_negative = self.w(p + 'ssm_a')
        dt_bias = self.w(p + 'ssm_dt')
        if a_negative.size != self.ssm_value_heads or dt_bias.size != self.ssm_value_heads:
            raise ValueError('Gated DeltaNet ssm_a/ssm_dt head count mismatch')
        # GGUF stores A as its already-negative exponential, not as A_log.
        decay = a_negative * _softplus(alpha_logits + dt_bias)
        beta = _sigmoid(beta_logits)
        state = self.recurrent_state[layer]
        if not self._delta_net_step_checked:
            self._delta_net_step_checked = True
            try:
                from .numba_kernels import delta_net_step
                self._delta_net_step_impl = delta_net_step
            except Exception:
                self._delta_net_step_impl = None
        if self._delta_net_step_impl is not None:
            try:
                core = self._delta_net_step_impl(state, q, k, v, decay, beta)
            except Exception:
                # Keep generation available if Numba is installed but JIT or
                # runtime compilation is unavailable on this machine.
                self._delta_net_step_impl = None
                state *= np.exp(decay)[:, None, None]
                predicted = np.einsum('hkv,hk->hv', state, k, optimize=True)
                delta = (v - predicted) * beta[:, None]
                state += k[:, :, None] * delta[:, None, :]
                core = np.einsum('hkv,hk->hv', state, q, optimize=True)
        else:
            state *= np.exp(decay)[:, None, None]
            predicted = np.einsum('hkv,hk->hv', state, k, optimize=True)
            delta = (v - predicted) * beta[:, None]
            state += k[:, :, None] * delta[:, None, :]
            core = np.einsum('hkv,hk->hv', state, q, optimize=True)
        norm_w = self.w(p + 'ssm_norm.weight')
        core = _rms(core, norm_w, self.eps) * _silu(z.reshape(self.ssm_value_heads, self.ssm_value_dim))
        return self.mat(core.reshape(self.dim), p + 'ssm_out.weight')

    def _rope_sequence(self, a, positions):
        """Apply split-half RoPE to [tokens, heads, head_dim] in one operation."""
        rotary = min(self.rope_dim, self.head_dim)
        rotary -= rotary % 2
        half = rotary // 2
        inv = 1.0 / (self.rope_base ** (np.arange(half, dtype=np.float32) / max(half, 1)))
        angle = np.asarray(positions, dtype=np.float32)[:, None] * inv[None, :]
        c, s = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]
        out = a.copy()
        left = a[..., :half].copy()
        right = a[..., half:rotary].copy()
        out[..., :half] = left * c - right * s
        out[..., half:rotary] = left * s + right * c
        return out

    def _full_attention_sequence(self, x, layer, positions, stop_event=None):
        p = f'blk.{layer}.'
        token_count = x.shape[0]
        qg = self.mat(x, p + 'attn_q.weight').reshape(token_count, self.n_heads, self.head_dim * 2)
        q = qg[:, :, :self.head_dim]
        gate = qg[:, :, self.head_dim:]
        k = self.mat(x, p + 'attn_k.weight').reshape(token_count, self.n_kv_heads, self.head_dim)
        v = self.mat(x, p + 'attn_v.weight').reshape(token_count, self.n_kv_heads, self.head_dim)
        q = _rms(q, self.w(p + 'attn_q_norm.weight'), self.eps)
        k = _rms(k, self.w(p + 'attn_k_norm.weight'), self.eps)
        q = self._rope_sequence(q, positions)
        k = self._rope_sequence(k, positions)
        self.k_cache[layer][positions] = k
        self.v_cache[layer][positions] = v
        group = self.n_heads // self.n_kv_heads
        att = np.empty((token_count, self.dim), dtype=np.float32)
        for idx, position in enumerate(positions):
            if stop_event is not None and stop_event.is_set():
                return None
            keys = self.k_cache[layer][:int(position) + 1]
            values = self.v_cache[layer][:int(position) + 1]
            q_grouped = q[idx].reshape(self.n_kv_heads, group, self.head_dim)
            scores = np.einsum('hgd,thd->hgt', q_grouped, keys, optimize=True) / math.sqrt(self.head_dim)
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            token_att = np.einsum('hgt,thd->hgd', probs, values, optimize=True).reshape(self.dim)
            att[idx] = token_att * _sigmoid(gate[idx].reshape(self.dim))
        return self.mat(att, p + 'attn_output.weight')

    def _linear_attention_sequence(self, x, layer, stop_event=None):
        p = f'blk.{layer}.'
        token_count = x.shape[0]
        qkv = self.mat(x, p + 'attn_qkv.weight')
        z = self.mat(x, p + 'attn_gate.weight')
        beta_logits = self.mat(x, p + 'ssm_beta.weight')
        alpha_logits = self.mat(x, p + 'ssm_alpha.weight')
        key_width = self.ssm_key_heads * self.ssm_key_dim
        value_width = self.ssm_value_heads * self.ssm_value_dim
        if qkv.shape[1] != key_width * 2 + value_width:
            raise ValueError(f'{p}attn_qkv.weight output width {qkv.shape[1]} is not expected {key_width * 2 + value_width}')
        conv_w = self.w(p + 'ssm_conv1d.weight')
        if conv_w.shape != (qkv.shape[1], self.conv_kernel):
            raise ValueError(f'{p}ssm_conv1d.weight shape {conv_w.shape} != {(qkv.shape[1], self.conv_kernel)}')
        mixed = np.empty_like(qkv)
        # Gated DeltaNet's convolution and recurrent state are causal, so only
        # this small state update remains sequential; all large projections are batched.
        for idx in range(token_count):
            if stop_event is not None and stop_event.is_set():
                return None
            raw = np.concatenate((self.conv_state[layer], qkv[idx][None, :]), axis=0)
            convolved = np.sum(raw.T * conv_w, axis=1)
            self.conv_state[layer] = raw[1:].copy()
            mixed[idx] = _silu(convolved)
        q_all, k_all, v_all = np.split(mixed, (key_width, key_width * 2), axis=1)
        q_all = q_all.reshape(token_count, self.ssm_key_heads, self.ssm_key_dim)
        k_all = k_all.reshape(token_count, self.ssm_key_heads, self.ssm_key_dim)
        v_all = v_all.reshape(token_count, self.ssm_value_heads, self.ssm_value_dim)
        q_all /= np.maximum(np.linalg.norm(q_all, axis=-1, keepdims=True), 1e-6)
        k_all /= np.maximum(np.linalg.norm(k_all, axis=-1, keepdims=True), 1e-6)
        q_all /= math.sqrt(self.ssm_key_dim)
        if self.ssm_value_heads % self.ssm_key_heads:
            raise ValueError('Gated DeltaNet value heads must be a multiple of key heads')
        repeats = self.ssm_value_heads // self.ssm_key_heads
        if bool(self.gguf.metadata.get(self.metadata_prefix + 'ssm.v_head_reordered', True)):
            head_map = np.tile(np.arange(self.ssm_key_heads), repeats)
        else:
            head_map = np.repeat(np.arange(self.ssm_key_heads), repeats)
        q_all = q_all[:, head_map]
        k_all = k_all[:, head_map]
        a_negative = self.w(p + 'ssm_a')
        dt_bias = self.w(p + 'ssm_dt')
        if a_negative.size != self.ssm_value_heads or dt_bias.size != self.ssm_value_heads:
            raise ValueError('Gated DeltaNet ssm_a/ssm_dt head count mismatch')
        beta_all = _sigmoid(beta_logits)
        core_all = np.empty((token_count, self.ssm_value_heads, self.ssm_value_dim), dtype=np.float32)
        state = self.recurrent_state[layer]
        for idx in range(token_count):
            if stop_event is not None and stop_event.is_set():
                return None
            decay = a_negative * _softplus(alpha_logits[idx] + dt_bias)
            state *= np.exp(decay)[:, None, None]
            predicted = np.einsum('hkv,hk->hv', state, k_all[idx], optimize=True)
            delta = (v_all[idx] - predicted) * beta_all[idx, :, None]
            state += k_all[idx, :, :, None] * delta[:, None, :]
            core = np.einsum('hkv,hk->hv', state, q_all[idx], optimize=True)
            core = _rms(core, self.w(p + 'ssm_norm.weight'), self.eps)
            core_all[idx] = core * _silu(z[idx].reshape(self.ssm_value_heads, self.ssm_value_dim))
        return self.mat(core_all.reshape(token_count, self.dim), p + 'ssm_out.weight')

    def forward_sequence(self, tokens, stop_event=None):
        """Batched causal prompt prefill; returns only the final token logits."""
        token_ids = list(tokens)
        if not token_ids:
            raise ValueError('forward_sequence requires at least one token')
        if len(token_ids) > self.max_seq:
            token_ids = token_ids[-self.max_seq:]
        positions = np.arange(len(token_ids), dtype=np.int64)
        emb = self.gguf.find('token_embd.weight')
        x = np.stack([self.gguf.tensor_row(emb, int(token)) for token in token_ids]).astype(np.float32, copy=False)
        for layer in range(self.n_layers):
            if stop_event is not None and stop_event.is_set():
                return None
            p = f'blk.{layer}.'
            residual = x
            x_norm = self.norm(x, p + 'attn_norm.weight')
            if layer in self.full_layers:
                attn = self._full_attention_sequence(x_norm, layer, positions, stop_event)
            else:
                attn = self._linear_attention_sequence(x_norm, layer, stop_event)
            if attn is None:
                return None
            x = residual + attn
            residual = x
            h = self.norm(x, p + 'post_attention_norm.weight')
            gate = _silu(self.mat(h, p + 'ffn_gate.weight'))
            up = self.mat(h, p + 'ffn_up.weight')
            x = residual + self.mat(gate * up, p + 'ffn_down.weight')
            if self.progress_callback is not None and (layer + 1) % 4 == 0:
                self.progress_callback(f'Qwen3.5 CPU · prompt prefill · layer {layer + 1}/{self.n_layers}')
        self.cache_length = len(token_ids)
        last = self.norm(x[-1], 'output_norm.weight')
        output = self.gguf.find('output.weight')
        return self.mat(last, output) if output else self.gguf.matvec(emb, last)

    def forward(self, token, position):
        if not 0 <= position < self.max_seq:
            raise ValueError(f'position {position} exceeds configured context {self.max_seq}')
        emb = self.gguf.find('token_embd.weight')
        x = self.gguf.tensor_row(emb, int(token)) if hasattr(self.gguf, 'tensor_row') else self.w(emb)[int(token)].copy()
        for layer in range(self.n_layers):
            p = f'blk.{layer}.'
            residual = x
            x_norm = self.norm(x, p + 'attn_norm.weight')
            if layer in self.full_layers:
                attn = self._full_attention(x_norm, layer, position)
            else:
                attn = self._linear_attention(x_norm, layer)
            x = residual + attn
            residual = x
            h = self.norm(x, p + 'post_attention_norm.weight')
            gate = _silu(self.mat(h, p + 'ffn_gate.weight'))
            up = self.mat(h, p + 'ffn_up.weight')
            x = residual + self.mat(gate * up, p + 'ffn_down.weight')
            if self.progress_callback is not None and (layer + 1) % 4 == 0:
                self.progress_callback(f'Qwen3.5 CPU · position {position + 1} · layer {layer + 1}/{self.n_layers}')
        x = self.norm(x, 'output_norm.weight')
        output = self.gguf.find('output.weight')
        if output:
            return self.mat(x, output)
        return self.gguf.matvec(emb, x)

    def generate(self, prompt_ids, tokenizer, stop_event, emit, temperature=.8, top_p=.95,
                 repetition_penalty=1.1, max_new=256):
        from .sampler import sample
        for cache in self.k_cache.values(): cache.fill(0)
        for cache in self.v_cache.values(): cache.fill(0)
        for cache in self.conv_state.values(): cache.fill(0)
        for cache in self.recurrent_state.values(): cache.fill(0)
        ids = list(prompt_ids) or [tokenizer.bos]
        prompt = ids[-self.max_seq:]
        self.cache_length = 0
        self.last_prefill_seconds = 0.0
        self.last_generated_tokens = 0
        prefill_started = time.perf_counter()
        if stop_event.is_set():
            return
        # Prefill all prompt tokens together. The model's recurrent state is
        # still updated causally, but each large projection reads its weights
        # once for the whole prompt instead of once per prompt token.
        logits = self.forward_sequence(prompt, stop_event)
        if logits is None or stop_event.is_set():
            return
        self.last_prefill_seconds = time.perf_counter() - prefill_started
        self.decode_profiler.start_decode()
        decoder = tokenizer.new_decoder()
        def flush():
            tail = decoder.decode(b'', final=True)
            decoder.reset()
            if tail: emit(tail)
        for generated_idx in range(max_new):
            if stop_event.is_set() or self.cache_length >= self.max_seq:
                flush(); return
            sample_started = time.perf_counter() if self.decode_profiler.enabled else 0.0
            nxt = sample(logits, ids, temperature, top_p, repetition_penalty)
            if self.decode_profiler.enabled:
                self.decode_profiler.record_sampling(time.perf_counter() - sample_started)
            if nxt in getattr(tokenizer, 'stop_ids', {tokenizer.eos}):
                flush(); return
            ids.append(nxt)
            self.last_generated_tokens += 1
            emit(tokenizer.decode_token(nxt, decoder))
            # Do not spend another full model pass computing logits after the
            # final requested token; on a 9B CPU model that is expensive.
            if generated_idx + 1 >= max_new:
                flush(); return
            if stop_event.is_set():
                flush(); return
            forward_started = time.perf_counter() if self.decode_profiler.enabled else 0.0
            logits = self.forward(nxt, self.cache_length)
            if self.decode_profiler.enabled:
                self.decode_profiler.record_forward(time.perf_counter() - forward_started)
            self.cache_length += 1
        flush()
