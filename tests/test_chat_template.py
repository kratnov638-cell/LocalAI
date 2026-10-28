import unittest

from app.window import format_chat


class ChatTemplateTests(unittest.TestCase):
    def test_qwen35_disables_thinking_for_user_facing_generation(self):
        metadata = {
            'tokenizer.ggml.pre': 'qwen35',
            'tokenizer.chat_template': "{{ '<think>\\n' }}",
        }
        prompt = format_chat([{'role': 'user', 'content': 'Что ты умеешь?'}], metadata)
        self.assertTrue(prompt.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'))
        self.assertIn('<|im_start|>user\nЧто ты умеешь?<|im_end|>\n', prompt)
        self.assertIn('Отвечай исключительно на русском языке', prompt)
        self.assertIn('не смешивай русский с другими языками', prompt)

    def test_qwen35_preserves_explicit_system_message(self):
        metadata = {
            'tokenizer.ggml.pre': 'qwen35',
            'tokenizer.chat_template': "{{ '<think>\\n' }}",
        }
        prompt = format_chat([
            {'role': 'system', 'content': 'Ты отвечаешь кратко.'},
            {'role': 'user', 'content': 'Привет'},
        ], metadata)
        self.assertEqual(prompt.count('<|im_start|>system\n'), 1)
        self.assertIn('<|im_start|>system\nТы отвечаешь кратко.<|im_end|>', prompt)

    def test_qwen35_uses_selected_english_language(self):
        metadata = {
            'tokenizer.ggml.pre': 'qwen35',
            'tokenizer.chat_template': "{{ '<think>\\n' }}",
            '_localai_language': 'en',
        }
        prompt = format_chat([{'role': 'user', 'content': 'What can you do?'}], metadata)
        self.assertIn('Answer exclusively in English', prompt)
        self.assertNotIn('Отвечай исключительно на русском языке', prompt)

    def test_llama3_template_includes_bos_from_gguf_metadata(self):
        tokens = ['x'] * 128001
        tokens[128000] = '<|begin_of_text|>'
        metadata = {
            'tokenizer.ggml.pre': 'llama-bpe',
            'tokenizer.ggml.bos_token_id': 128000,
            'tokenizer.ggml.tokens': tokens,
            'tokenizer.chat_template': "{{ bos_token }}",
        }
        prompt = format_chat([{'role': 'user', 'content': 'Say hello.'}], metadata)
        self.assertTrue(prompt.startswith('<|begin_of_text|><|start_header_id|>user<|end_header_id|>'))
        self.assertIn('Say hello.<|eot_id|>', prompt)
        self.assertTrue(prompt.endswith('<|start_header_id|>assistant<|end_header_id|>\n\n'))

    def test_qwen3_without_thinking_template_keeps_existing_prefix(self):
        metadata = {'tokenizer.ggml.pre': 'qwen3', 'tokenizer.chat_template': ''}
        prompt = format_chat([{'role': 'user', 'content': 'Привет'}], metadata)
        self.assertTrue(prompt.endswith('<|im_start|>assistant\n'))
        self.assertFalse(prompt.endswith('<think>\n'))

    def test_zaya_closes_think_channel_for_user_facing_answer(self):
        metadata = {
            'general.architecture': 'zaya',
            'tokenizer.chat_template': "{{ bos_token }}<|im_start|>assistant\\n<think>\\n",
            'tokenizer.ggml.model': 'gemma4',
        }
        prompt = format_chat([{'role': 'user', 'content': 'Write a Python function.'}], metadata)
        self.assertIn('<|im_start|>user\nWrite a Python function.<|im_end|>\n', prompt)
        self.assertTrue(prompt.endswith('<|im_start|>assistant\n<think>\n</think>\n\n'))
        self.assertNotIn('Отвечай исключительно на русском', prompt)
        self.assertTrue(prompt.startswith('<bos><|im_start|>system\n'))
        self.assertIn('Отвечай по-русски естественно и грамотно.', prompt)


if __name__ == '__main__':
    unittest.main()
