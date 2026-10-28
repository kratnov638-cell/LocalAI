"""Qwen3.5 GGUF architecture inspection and hybrid layer-layout validation.

This module records the hybrid layer layout from GGUF metadata and tensor
names. Keeping inspection separate from execution lets the loader validate the
hybrid layout before constructing the dedicated NumPy text decoder.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen35Layer:
    index: int
    kind: str  # 'full_attention' or 'linear_attention'


@dataclass(frozen=True)
class Qwen35Layout:
    hidden_size: int
    layer_count: int
    full_attention_interval: int
    vocab_size: int
    layers: tuple[Qwen35Layer, ...]
    has_vision_tower: bool

    @classmethod
    def inspect(cls, metadata: dict, tensors: dict) -> 'Qwen35Layout':
        architecture = str(metadata.get('general.architecture', '')).lower()
        if architecture not in {'qwen35', 'qwen3.5'}:
            raise ValueError(f'Qwen35Layout requires qwen35 architecture, got {architecture!r}')
        prefix = 'qwen35.' if any(key.startswith('qwen35.') for key in metadata) else 'qwen3.5.'
        def required(name):
            key = prefix + name
            if key not in metadata:
                raise ValueError(f'Missing required Qwen3.5 metadata: {key}')
            return int(metadata[key])
        hidden = required('embedding_length')
        count = required('block_count')
        interval = required('full_attention_interval')
        if hidden <= 0 or count <= 0 or interval <= 0:
            raise ValueError('Qwen3.5 dimensions and attention interval must be positive')
        vocab = len(metadata.get('tokenizer.ggml.tokens', []))
        if vocab <= 0:
            raise ValueError('Qwen3.5 GGUF has no tokenizer.ggml.tokens vocabulary')
        layers = []
        for i in range(count):
            # Qwen3.5 GGUFs expose the hybrid pattern directly through tensor
            # presence: full-attention blocks have attn_q/k/v, while GDN blocks
            # have the recurrent SSM parameter set and fused attn_qkv.
            names = {f'blk.{i}.{suffix}' for suffix in (
                'attn_q.weight', 'attn_k.weight', 'attn_v.weight',
                'attn_qkv.weight', 'ssm_a', 'ssm_conv1d.weight', 'ssm_out.weight')}
            present = names.intersection(tensors)
            is_full = f'blk.{i}.attn_q.weight' in present
            is_linear = f'blk.{i}.ssm_a' in present and f'blk.{i}.attn_qkv.weight' in present
            if is_full == is_linear:
                raise ValueError(f'Qwen3.5 layer {i} has ambiguous/incomplete attention tensors')
            expected_full = ((i + 1) % interval == 0)
            if is_full != expected_full:
                raise ValueError(
                    f'Qwen3.5 layer {i} tensor layout conflicts with full_attention_interval={interval}'
                )
            layers.append(Qwen35Layer(i, 'full_attention' if is_full else 'linear_attention'))
        return cls(
            hidden_size=hidden, layer_count=count, full_attention_interval=interval,
            vocab_size=vocab, layers=tuple(layers),
            has_vision_tower=any(name.startswith('v.') for name in tensors),
        )

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        return tuple(layer.index for layer in self.layers if layer.kind == 'full_attention')

    @property
    def linear_attention_layers(self) -> tuple[int, ...]:
        return tuple(layer.index for layer in self.layers if layer.kind == 'linear_attention')
