import unittest

from engine.architectures import get_architecture, register_architecture, registered_architectures
from engine.model import LlamaModel, Qwen2Model, Qwen3Model
from engine.qwen35 import Qwen35Model


class ArchitectureRegistryTests(unittest.TestCase):
    def test_existing_architectures_are_explicit(self):
        self.assertIs(get_architecture('llama'), LlamaModel)
        self.assertIs(get_architecture('mistral'), LlamaModel)
        self.assertIs(get_architecture('qwen2'), Qwen2Model)
        self.assertIs(get_architecture('qwen3'), Qwen3Model)
        self.assertIs(get_architecture('qwen35'), Qwen35Model)
        self.assertIs(get_architecture('qwen3.5'), Qwen35Model)
        self.assertEqual(registered_architectures(), ('llama', 'mistral', 'qwen2', 'qwen3', 'qwen3.5', 'qwen35'))

    def test_unknown_architecture_does_not_fall_back_to_llama(self):
        with self.assertRaisesRegex(NotImplementedError, 'not implemented'):
            get_architecture('future-family')

    def test_registry_can_add_a_separate_implementation(self):
        class DummyModel:
            pass
        register_architecture('test-family', DummyModel)
        try:
            self.assertIs(get_architecture('test-family'), DummyModel)
        finally:
            # Avoid leaking test registration to other tests.
            from engine import architectures
            architectures._ARCHITECTURES.pop('test-family', None)

    def test_duplicate_registration_requires_explicit_replace(self):
        with self.assertRaisesRegex(ValueError, 'already registered'):
            register_architecture('llama', LlamaModel)
