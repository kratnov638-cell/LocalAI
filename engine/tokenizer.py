"""Small, dependency-free byte-level BPE tokenizer for GGUF vocabularies.

This intentionally implements tokenization itself; it does not delegate to a
third-party inference/tokenizer runtime. GGUF's tokenizer metadata determines
the vocabulary, merge ranks, BOS/EOS IDs and pre-tokenizer family.
"""
import json
import re
import codecs
from pathlib import Path

class BPETokenizer:
    def __init__(self, metadata, model_path=None):
        tokens=metadata.get('tokenizer.ggml.tokens')
        if not tokens:
            hint=' Put tokenizer.json next to the GGUF.'
            p=Path(model_path).with_name('tokenizer.json') if model_path else None
            if p and p.exists():
                data=json.loads(p.read_text(encoding='utf-8')); self._from_json(data); return
            raise ValueError('GGUF has no tokenizer.ggml.tokens.'+hint)
        self.tokens=list(tokens); self.vocab={s:i for i,s in enumerate(self.tokens)}
        merges=metadata.get('tokenizer.ggml.merges',[]); self.ranks={tuple(x.split(' ',1)):i for i,x in enumerate(merges) if ' ' in x}
        self.bos=int(metadata.get('tokenizer.ggml.bos_token_id',1)); self.eos=int(metadata.get('tokenizer.ggml.eos_token_id',2))
        self.add_bos=bool(metadata.get('tokenizer.ggml.add_bos_token',False))
        self.pre=str(metadata.get('tokenizer.ggml.pre','default'))
        self.model_type=str(metadata.get('tokenizer.ggml.model','')).lower()
        self._init_byte_alphabet()
        self.special_tokens = sorted(
            ((token, i) for i, token in enumerate(self.tokens)
             if token and token.startswith('<') and token.endswith('>')),
            key=lambda item: len(item[0]), reverse=True)
        self._special_set = {token for token, _ in self.special_tokens}
        self._special_re = re.compile('|'.join(re.escape(t) for t, _ in self.special_tokens)) if self.special_tokens else None
        self.stop_ids = {self.eos}
        for key in ('tokenizer.ggml.eot_token_id', 'tokenizer.ggml.eom_token_id'):
            if key in metadata:
                self.stop_ids.add(int(metadata[key]))
        self.stop_ids.update(token_id for token, token_id in self.special_tokens
                             if token in {'<|eot_id|>', '<|im_end|>', '<|fim_suffix|>', '<|end_of_turn|>'})
    def _init_byte_alphabet(self):
        # GPT-2's reversible bytes_to_unicode() alphabet. All 256 byte values
        # map to printable Unicode code points, including bytes absent from
        # the original visible-character ranges.
        bs=list(range(ord('!'),ord('~')+1))+list(range(ord('¡'),ord('¬')+1))+list(range(ord('®'),ord('ÿ')+1))
        cs=bs[:]; extra=0
        for b in range(256):
            if b not in bs:
                bs.append(b); cs.append(256+extra); extra+=1
        self.byte_encoder=dict(zip(bs,map(chr,cs)))
        self.byte_decoder={v:k for k,v in self.byte_encoder.items()}
    def _from_json(self,data):
        model=data.get('model',{}); vocab=model.get('vocab',{}); self.tokens=[None]*(max(vocab.values(),default=-1)+1)
        for s,i in vocab.items(): self.tokens[i]=s
        self.vocab=vocab; self.ranks={tuple(m.split(' ',1)):i for i,m in enumerate(model.get('merges',[])) if ' ' in m}; self.bos=1; self.eos=2; self.add_bos=False; self.pre='default'; self.model_type=str(model.get('type','')).lower(); self._init_byte_alphabet()
        self.special_tokens=sorted(((token,i) for token,i in vocab.items() if token and token.startswith('<') and token.endswith('>')),key=lambda item:len(item[0]),reverse=True)
        self._special_set={token for token,_ in self.special_tokens}; self.stop_ids={self.eos}
        self.stop_ids.update(token_id for token, token_id in self.special_tokens
                             if token in {'<|eot_id|>', '<|im_end|>', '<|fim_suffix|>', '<|end_of_turn|>'})
        self._special_re=re.compile('|'.join(re.escape(t) for t,_ in self.special_tokens)) if self.special_tokens else None
    def _bpe(self,word):
        parts=list(word)
        while len(parts)>1:
            pairs={(parts[i],parts[i+1]) for i in range(len(parts)-1)}; pair=min(pairs,key=lambda p:self.ranks.get(p,10**18))
            if pair not in self.ranks:break
            merged=[]; i=0
            while i<len(parts):
                if i<len(parts)-1 and (parts[i],parts[i+1])==pair: merged.append(parts[i]+parts[i+1]); i+=2
                else: merged.append(parts[i]); i+=1
            parts=merged
        return parts
    def _pieces(self,text):
        # Llama 3's tiktoken pattern allows one non-letter/number character
        # before a letter run (e.g. the hyphen in "-word"). Python re lacks
        # Unicode \p{L}/\p{N}, so use Unicode-aware classes and preserve the
        # underscore exception explicitly. Qwen's pre-tokenizer is kept separate.
        contractions = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        if self.pre.startswith(('llama-bpe', 'llama3')):
            pattern = contractions + r"|(?:[^\r\n\w]|_)?[^\W\d_]+|\d{1,3}| ?(?:[^\s\w]|_)+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        elif self.pre.startswith(('qwen2', 'qwen3', 'qwen35')):
            pattern = contractions + r"| ?[^\W\d_]+|\d{1,3}| ?[^\w\s]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        else:
            pattern = r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\w\s]+|\s+(?!\S)|\s+"
        return re.findall(pattern, text)
    def _encode_plain(self,text):
        ids=[]
        # Gemma4 GGUF vocabularies are SentencePiece-style Unicode BPE: spaces
        # are represented by U+2581 and token strings are not GPT-2 byte-alphabet
        # encodings. Feeding them through byte_encoder silently creates wrong
        # token IDs (or fails on perfectly ordinary input).
        if self.model_type == 'gemma4':
            normalized=text.replace(' ', '\u2581')
            for token in self._bpe(normalized):
                if token not in self.vocab:
                    raise ValueError(f'Gemma4 tokenizer vocabulary lacks token {token!r}')
                ids.append(self.vocab[token])
            return ids
        for piece in self._pieces(text):
            encoded=''.join(self.byte_encoder[b] for b in piece.encode('utf-8'))
            for token in self._bpe(encoded):
                if token not in self.vocab: raise ValueError(f'Tokenizer vocabulary lacks token {token!r}')
                ids.append(self.vocab[token])
        return ids
    def encode(self,text,add_bos=None):
        ids=[]; use_bos=self.add_bos if add_bos is None else add_bos
        if use_bos: ids.append(self.bos)
        # Preserve registered special-token IDs instead of byte-encoding them.
        if self._special_re:
            chunks=self._special_re.split(text); specials=self._special_re.findall(text)
            for i,chunk in enumerate(chunks):
                if chunk: ids.extend(self._encode_plain(chunk))
                if i<len(specials): ids.append(self.vocab[specials[i]])
        else:
            ids.extend(self._encode_plain(text))
        return ids
    def decode(self,ids):
        if getattr(self, 'model_type', '') == 'gemma4':
            pieces=[]
            for idx in ids:
                if not 0 <= int(idx) < len(self.tokens):
                    continue
                token=self.tokens[int(idx)] or ''
                if token in self._special_set:
                    # Preserve meaningful chat markers, but don't render the
                    # ordinary BOS/EOS control tokens as user-facing text.
                    if token not in {'<bos>', '<eos>', '<pad>', '<unk>'}:
                        pieces.append(token)
                else:
                    pieces.append(token)
            return ''.join(pieces).replace('\u2581', ' ')
        result=[]; byte_chars=[]
        def flush():
            if byte_chars:
                result.append(bytes(self.byte_decoder[c] for c in byte_chars).decode('utf-8',errors='replace'))
                byte_chars.clear()
        for idx in ids:
            if not 0<=int(idx)<len(self.tokens):
                continue
            token=self.tokens[int(idx)] or ''
            if token in self._special_set:
                flush(); result.append(token)
            else:
                byte_chars.extend(token)
        flush()
        return ''.join(result)

    def new_decoder(self):
        """Return a stateful UTF-8 decoder for token-by-token generation."""
        return codecs.getincrementaldecoder('utf-8')(errors='replace')

    def decode_token(self, token_id, decoder):
        """Decode one BPE token without corrupting multi-token UTF-8 characters."""
        if not 0 <= int(token_id) < len(self.tokens):
            return ''
        token = self.tokens[int(token_id)] or ''
        if getattr(self, 'model_type', '') == 'gemma4':
            if token in {'<bos>', '<eos>', '<pad>', '<unk>'}:
                return ''
            return token.replace('\u2581', ' ')
        if token in self._special_set:
            tail = decoder.decode(b'', final=True)
            decoder.reset()
            return tail + token
        try:
            raw = bytes(self.byte_decoder[char] for char in token)
        except KeyError:
            tail = decoder.decode(b'', final=True)
            decoder.reset()
            return tail + token
        return decoder.decode(raw, final=False)
