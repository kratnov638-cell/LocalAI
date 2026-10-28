import unittest

from app.window import (
    append_user_message, format_chat, generation_settings_for_model, is_followup_request,
    is_code_only_request, is_single_function_request, prepare_generated_piece, summarize_answer_for_context,
    truncate_repeated_function_definition,
)


class ChatTemplateTests(unittest.TestCase):
    def test_resending_same_last_user_prompt_replaces_previous_turn(self):
        content = 'Сделай на python функцию is_year_leap. Пиши только код'
        messages = [
            {'role': 'user', 'content': content},
            {'role': 'assistant', 'content': 'пп'},
        ]
        append_user_message(messages, content)
        self.assertEqual(messages, [{'role': 'user', 'content': content}])

    def test_distinct_zaya_prompt_starts_fresh_context(self):
        messages = [
            {'role': 'user', 'content': 'Первый вопрос'},
            {'role': 'assistant', 'content': 'Ответ'},
        ]
        append_user_message(messages, 'Напиши сортировку списка', 'zaya')
        self.assertEqual(messages, [{'role': 'user', 'content': 'Напиши сортировку списка'}])

    def test_zaya_followup_keeps_answer_as_context_not_previous_user_prompt(self):
        messages = [
            {'role': 'user', 'content': 'Старое несвязанное задание'},
            {'role': 'assistant', 'content': 'Старый ответ'},
            {'role': 'user', 'content': 'Сделай функцию високосного года. Пиши только код'},
            {'role': 'assistant', 'content': 'def is_year_leap(year): return year % 4 == 0'},
        ]
        append_user_message(messages, 'теперь напиши код проверки', 'zaya')
        self.assertEqual([m['role'] for m in messages], ['user'])
        self.assertIn('is_year_leap(year)', messages[0]['content'])
        self.assertIn('assert is_year_leap(2000) is True', messages[0]['content'])
        self.assertIn('assert is_year_leap(1900) is False', messages[0]['content'])
        self.assertNotIn('return year % 4 == 0', messages[0]['content'])
        prompt = format_chat(messages, {
            'general.architecture': 'zaya',
            'tokenizer.chat_template': "{{ '<|im_start|>' }}",
        })
        self.assertNotIn('теперь напиши код проверки', prompt)
        self.assertEqual(prompt.count('is_year_leap(year)'), 1)
        self.assertNotIn('Сделай функцию високосного года', prompt)
        self.assertEqual(prompt.count('<|im_start|>user\n'), 1)

    def test_context_summary_keeps_function_name_not_implementation(self):
        summary = summarize_answer_for_context(
            '```python\ndef is_year_leap(year):\n    return year % 4 == 0\n```'
        )
        self.assertIn('is_year_leap(year)', summary)
        self.assertNotIn('return year % 4 == 0', summary)

    def test_zaya_followup_detection_does_not_match_independent_task(self):
        self.assertTrue(is_followup_request('теперь напиши код проверки'))
        self.assertTrue(is_followup_request('Напиши функцию для проверки'))
        self.assertTrue(is_followup_request('Исправь ошибку в этой функции'))
        self.assertFalse(is_followup_request('Напиши функцию сортировки списка'))

    def test_zaya_function_for_checking_followup_becomes_tests_for_previous_function(self):
        messages = [
            {'role': 'user', 'content': 'Напиши функцию is_year_leap'},
            {'role': 'assistant', 'content': '```python\ndef is_year_leap(year):\n    return year % 4 == 0\n```'},
        ]
        append_user_message(messages, 'Напиши функцию для проверки', 'zaya')
        self.assertEqual([m['role'] for m in messages], ['user'])
        self.assertIn('assert is_year_leap(2000) is True', messages[0]['content'])
        self.assertIn('assert is_year_leap(1900) is False', messages[0]['content'])
        self.assertNotIn('def is_year_leap', messages[0]['content'])

    def test_resending_clarified_zaya_followup_retries_without_stale_context(self):
        messages = [
            {'role': 'user', 'content': 'def is_year_leap(year): return True'},
            {'role': 'assistant', 'content': 'Ответ с кодом'},
        ]
        request = 'теперь напиши код проверки'
        append_user_message(messages, request, 'zaya')
        messages.append({'role': 'assistant', 'content': 'тестовый ответ'})
        append_user_message(messages, request, 'zaya')
        self.assertEqual(messages, [{'role': 'user', 'content': request}])

    def test_zaya_generation_defaults_use_neutral_repetition_penalty(self):
        got = generation_settings_for_model(
            {'temperature': 0.6, 'top_p': 0.95, 'max_new': 128, 'language': 'ru'},
            'zaya',
        )
        self.assertEqual(got['repetition_penalty'], 1.0)
        self.assertNotIn('language', got)

    def test_explicit_zaya_repetition_penalty_is_preserved(self):
        got = generation_settings_for_model(
            {'temperature': 0.4, 'repetition_penalty': 1.08}, 'zaya'
        )
        self.assertEqual(got['repetition_penalty'], 1.08)

    def test_code_only_request_detects_russian_and_english_instructions(self):
        self.assertTrue(is_code_only_request('Пиши только код'))
        self.assertTrue(is_code_only_request('Output code only'))
        self.assertFalse(is_code_only_request('Explain the algorithm'))

    def test_single_function_request_stops_after_one_complete_code_block(self):
        self.assertTrue(is_single_function_request('Написать функцию is_year_leap на Python'))
        self.assertTrue(is_single_function_request('Write a Python function to check leap years'))
        self.assertFalse(is_single_function_request('Explain the algorithm'))
        self.assertFalse(is_single_function_request('Напиши функцию и предложи несколько вариантов'))

    def test_single_function_request_truncates_after_first_complete_fence(self):
        first = '```python\ndef is_year_leap(year):\n    return True\n```'
        previous, visible, stop = prepare_generated_piece('', first + '\n\n```python\ndef other():', True)
        self.assertEqual(previous, first)
        self.assertEqual(visible, first)
        self.assertTrue(stop)

    def test_code_only_generation_stops_after_first_closed_code_fence(self):
        previous = '```python\ndef answer():\n    return 42\n'
        combined, visible, stop = prepare_generated_piece(previous, '```\n```python\nextra()', True)
        self.assertEqual(combined, '```python\ndef answer():\n    return 42\n```')
        self.assertEqual(visible, '```')
        self.assertTrue(stop)

    def test_non_code_generation_does_not_stop_at_code_fence(self):
        previous, visible, stop = prepare_generated_piece('', '```python', False)
        self.assertEqual(previous, '```python')
        self.assertEqual(visible, '```python')
        self.assertFalse(stop)

    def test_repeated_python_function_is_truncated_before_second_definition(self):
        text = (
            '```python\ndef is_year_leap(year):\n    return True\n```\n\n'
            '```python\ndef other(year):\n    return False\n```\n\n'
            '```python\ndef is_year_leap(year):\n    return False\n```'
        )
        result = truncate_repeated_function_definition(text)
        self.assertIsNotNone(result)
        self.assertIn('def other(year)', result)
        self.assertTrue(result.endswith('```'))
        self.assertEqual(result.count('```python'), 2)

    def test_distinct_python_functions_are_not_truncated(self):
        text = 'def first():\n    return 1\n\ndef second():\n    return 2'
        self.assertIsNone(truncate_repeated_function_definition(text))

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
        self.assertIn('Отвечай по-русски естественно.', prompt)


if __name__ == '__main__':
    unittest.main()
