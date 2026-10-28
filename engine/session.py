"""Persistent model session shared by successive chat turns."""
from pathlib import Path

from .architectures import get_architecture
from .gguf_parser import GGUFFile
from .tokenizer import BPETokenizer


class InferenceSession:
    """Own the GGUF mmap, tokenizer and decoded-weight cache for one model."""

    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self.gguf = GGUFFile(self.path)
        try:
            architecture = str(self.gguf.metadata.get('general.architecture', 'llama')).lower()
            model_class = get_architecture(architecture)
            self.tokenizer = BPETokenizer(self.gguf.metadata, self.path)
            self.model = model_class(self.gguf)
        except Exception:
            self.gguf.close()
            raise
        self.closed = False

    def close(self):
        if self.closed:
            return
        self.model._weights.clear()
        self.gguf.close()
        self.closed = True
