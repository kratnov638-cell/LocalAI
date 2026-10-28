import unittest

from engine.tokenizer import BPETokenizer


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        # All code points used by GPT-2's reversible byte alphabet are below
        # 512. This synthetic vocabulary isolates byte mapping/UTF-8 logic.
        tokens = [chr(i) for i in range(512)] + ['<|im_start|>', '<|im_end|>']
        self.tok = BPETokenizer({
            'tokenizer.ggml.tokens': tokens,
            'tokenizer.ggml.merges': [],
            'tokenizer.ggml.bos_token_id': 1,
            'tokenizer.ggml.eos_token_id': 2,
            'tokenizer.ggml.add_bos_token': False,
            'tokenizer.ggml.pre': 'qwen2',
        })

    def test_utf8_round_trip(self):
        text = 'Hello, мир! 123\n'
        ids = self.tok.encode(text)
        self.assertEqual(self.tok.decode(ids), text)
        decoder = self.tok.new_decoder()
        pieces = [self.tok.decode_token(token_id, decoder) for token_id in ids]
        pieces.append(decoder.decode(b'', final=True))
        self.assertEqual(''.join(pieces), text)

    def test_special_tokens_keep_single_id(self):
        ids = self.tok.encode('<|im_start|>assistant<|im_end|>')
        self.assertEqual(ids[0], self.tok.vocab['<|im_start|>'])
        self.assertEqual(ids[-1], self.tok.vocab['<|im_end|>'])
        self.assertIn(self.tok.vocab['<|im_end|>'], self.tok.stop_ids)
        self.assertEqual(self.tok.decode(ids), '<|im_start|>assistant<|im_end|>')
        decoder = self.tok.new_decoder()
        decoded = ''.join(self.tok.decode_token(token_id, decoder) for token_id in ids)
        decoded += decoder.decode(b'', final=True)
        self.assertEqual(decoded, '<|im_start|>assistant<|im_end|>')

    def test_llama3_pretokenizer_keeps_prefix_character_with_letter_run(self):
        llama3 = BPETokenizer({
            'tokenizer.ggml.tokens': self.tok.tokens,
            'tokenizer.ggml.merges': [],
            'tokenizer.ggml.pre': 'llama-bpe',
        })
        self.assertEqual(llama3._pieces('\u043f\u043e-\u0440\u0443\u0441\u0441\u043a\u0438'), ['\u043f\u043e', '-\u0440\u0443\u0441\u0441\u043a\u0438'])
        self.assertEqual(llama3._pieces('ab-cdefgh'), ['ab', '-cdefgh'])

    def test_qwen35_uses_qwen_bpe_pretokenization(self):
        qwen35 = BPETokenizer({
            'tokenizer.ggml.tokens': self.tok.tokens,
            'tokenizer.ggml.merges': [],
            'tokenizer.ggml.pre': 'qwen35',
        })
        self.assertEqual(qwen35._pieces('Привет 123!'), self.tok._pieces('Привет 123!'))

    def test_gemma4_uses_unicode_sentencepiece_bpe_not_gpt2_byte_encoding(self):
        tokens = ['<pad>', '<bos>', '<eos>', '<unk>', 'hello', '\u2581world', '!']
        merges = ['h e', 'he l', 'hel l', 'hell o', '\u2581 w', '\u2581w o', '\u2581wo r', '\u2581wor l', '\u2581worl d']
        tok = BPETokenizer({
            'tokenizer.ggml.tokens': tokens,
            'tokenizer.ggml.merges': merges,
            'tokenizer.ggml.model': 'gemma4',
            'tokenizer.ggml.bos_token_id': 1,
            'tokenizer.ggml.eos_token_id': 2,
            'tokenizer.ggml.add_bos_token': False,
        })
        ids = tok.encode('hello world!')
        self.assertEqual(ids, [tokens.index('hello'), tokens.index('\u2581world'), tokens.index('!')])
        self.assertEqual(tok.decode(ids), 'hello world!')
        self.assertEqual(''.join(tok.decode_token(i, None) for i in ids), 'hello world!')


if __name__ == '__main__':
    unittest.main()
