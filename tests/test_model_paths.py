import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.model_paths import default_models_dir, discover_gguf_models


class ModelPathTests(unittest.TestCase):
    def test_environment_override_wins_and_is_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = Path(tmp) / 'external models'
            with patch.dict(os.environ, {'LOCALAI_MODELS_DIR': str(override)}):
                self.assertEqual(default_models_dir(tmp), override.resolve())

    def test_missing_override_directory_returns_empty_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = Path(tmp) / 'missing'
            with patch.dict(os.environ, {'LOCALAI_MODELS_DIR': str(override)}):
                self.assertEqual(discover_gguf_models(tmp), [])

    def test_discovers_only_top_level_gguf_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'b.gguf').touch()
            (root / 'a.GGUF').touch()
            (root / 'ignore.bin').touch()
            (root / 'nested').mkdir()
            (root / 'nested' / 'hidden.gguf').touch()
            with patch.dict(os.environ, {'LOCALAI_MODELS_DIR': str(root)}):
                self.assertEqual([p.name for p in discover_gguf_models(root)], ['a.GGUF', 'b.gguf'])


if __name__ == '__main__':
    unittest.main()
