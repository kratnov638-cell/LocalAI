"""Architecture dispatch for the from-scratch GGUF inference engine.

Shared GGUF parsing, quantization and tokenization stay architecture-neutral;
model-family-specific forward passes are registered here.
"""
from __future__ import annotations

from typing import Type

from .model import LlamaModel, Qwen2Model, Qwen3Model
from .qwen35 import Qwen35Model
from .zaya import ZayaModel


# Keep dispatch explicit: Mistral uses the dense Llama-style core for the
# supported context window, Qwen2/Qwen3 have architecture-specific classes,
# and Qwen3.5 has its own hybrid attention/SSM implementation. Mixtral/MoE is
# not aliased to the dense Mistral path.
_ARCHITECTURES: dict[str, type] = {
    'llama': LlamaModel,
    'mistral': LlamaModel,
    'qwen2': Qwen2Model,
    'qwen3': Qwen3Model,
    'qwen35': Qwen35Model,
    'qwen3.5': Qwen35Model,
    'zaya': ZayaModel,
}


def register_architecture(name: str, model_class: type, *, replace: bool = False) -> None:
    """Register a model implementation for a GGUF general.architecture value."""
    key = name.strip().lower()
    if not key:
        raise ValueError('Architecture name cannot be empty')
    if not isinstance(model_class, type):
        raise TypeError('model_class must be a class')
    if key in _ARCHITECTURES and not replace:
        raise ValueError(f'Architecture {key!r} is already registered')
    _ARCHITECTURES[key] = model_class


def get_architecture(name: str) -> type:
    """Return the registered implementation or explain the unsupported family."""
    key = (name or 'llama').strip().lower()
    try:
        return _ARCHITECTURES[key]
    except KeyError:
        supported = ', '.join(sorted(_ARCHITECTURES))
        detail = ''
        raise NotImplementedError(
            f'GGUF architecture {key!r} is not implemented by this engine. '
            f'Registered architectures: {supported}.{detail}'
        ) from None


def registered_architectures() -> tuple[str, ...]:
    return tuple(sorted(_ARCHITECTURES))
