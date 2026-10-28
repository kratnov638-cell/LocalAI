"""Llama-family decoder. Uses NumPy/BLAS only; no llama.cpp/ggml bindings."""
import math, time, numpy as np
from .dequant import dequantize
from .decode_profiler import DecodeProfiler

class KVCache:
    def __init__(self,n_layers,n_kv_heads,head_dim,max_seq,dtype=np.float32):
        self.k=np.zeros((n_layers,max_seq,n_kv_heads,head_dim),dtype=dtype); self.v=np.zeros_like(self.k); self.length=0; self.max_seq=max_seq
    def reset(self):self.length=0

class LlamaModel:
    def __init__(self,gguf,max_seq=2048):
        self.gguf=gguf; m=gguf.metadata
        def meta(*keys,default=None):
            for k in keys:
                if k in m:return m[k]
            return default
        arch=meta('general.architecture',default='llama'); self.arch=arch
        self.n_layers=int(meta(f'{arch}.block_count','llama.block_count',default=0)); self.n_heads=int(meta(f'{arch}.attention.head_count','llama.attention.head_count',default=0)); self.n_kv_heads=int(meta(f'{arch}.attention.head_count_kv','llama.attention.head_count_kv',default=self.n_heads))
        self.dim=int(meta(f'{arch}.embedding_length','llama.embedding_length',default=0)); self.ffn=int(meta(f'{arch}.feed_forward_length','llama.feed_forward_length',default=0)); self.vocab=int(meta('tokenizer.ggml.vocab_size',default=len(m.get('tokenizer.ggml.tokens',[]))))
        self.eps=float(meta(f'{arch}.attention.layer_norm_rms_epsilon','llama.attention.layer_norm_rms_epsilon',default=1e-5)); self.rope_base=float(meta(f'{arch}.rope.freq_base','llama.rope.freq_base',default=1000000.0 if arch in {'qwen2', 'qwen3'} else 10000.0))
        self.rope_dim=int(meta(f'{arch}.rope.dimension_count','llama.rope.dimension_count',default=self.dim//self.n_heads if self.n_heads else 0))
        self.rope_scaling_type=str(meta(f'{arch}.rope.scaling.type','llama.rope.scaling.type',default='none')).lower()
        self.rope_scaling_factor=float(meta(f'{arch}.rope.scaling.factor','llama.rope.scaling.factor',default=1.0))
        self.rope_original_context=int(meta(f'{arch}.rope.scaling.original_context_length','llama.rope.scaling.original_context_length',default=8192))
        self.rope_low_freq=float(meta(f'{arch}.rope.scaling.low_freq_factor','llama.rope.scaling.low_freq_factor',default=1.0))
        self.rope_high_freq=float(meta(f'{arch}.rope.scaling.high_freq_factor','llama.rope.scaling.high_freq_factor',default=4.0))
        self.sliding_window=int(meta(f'{arch}.attention.sliding_window',default=0) or 0)
        if not all((self.n_layers,self.n_heads,self.n_kv_heads,self.dim,self.ffn,self.vocab)): raise ValueError(f'Missing required Llama architecture metadata for {arch!r}')
        if self.dim%self.n_heads:raise ValueError('embedding_length must be divisible by attention.head_count')
        self.head_dim=self.dim//self.n_heads; self.cache=KVCache(self.n_layers,self.n_kv_heads,self.head_dim,max_seq)
        self._weights={}
        self.decode_profiler = DecodeProfiler()
    def w(self,name):
        if name not in self._weights:self._weights[name]=self.gguf.tensor(name)
        return self._weights[name]
    def find(self,layer,suffix):
        candidates=[f'blk.{layer}.{suffix}',f'model.layers.{layer}.{suffix}']
        for c in candidates:
            if c in self.gguf.tensors:return c
        raise KeyError(f'Missing tensor for layer {layer}: {suffix}')
    def mat(self, x, name):
        profiler = getattr(self, 'decode_profiler', None)
        if profiler is None or not profiler.enabled or not profiler.active:
            return self._mat_impl(x, name)
        started = time.perf_counter()
        try:
            return self._mat_impl(x, name)
        finally:
            profiler.record_mat(name, time.perf_counter() - started)

    def _mat_impl(self,x,name):
        # GGUF logical shape is [out,in]; NumPy matmul expects [in,out].
        info = self.gguf.tensors.get(name) if hasattr(self.gguf, 'tensors') else None
        if info is not None and len(info.shape) != 2:
            raise ValueError(f'{name}: expected matrix, got {info.shape}')
        if info is not None and x.ndim == 1 and info.shape[1] == x.shape[-1] and hasattr(self.gguf, 'matvec'):
            # Quantized row-chunk matvec bounds peak RAM instead of caching the
            # entire model as float32 (which would exceed RAM for 9B weights).
            y = self.gguf.matvec(name, x)
        elif info is not None and x.ndim == 2 and info.shape[1] == x.shape[-1] and hasattr(self.gguf, 'matmul'):
            # Prompt prefill shares each packed weight tile across all prompt
            # positions; decoding the same GGUF matrix once per token is a major
            # avoidable cost on CPU.
            y = self.gguf.matmul(name, x)
        else:
            w=self.w(name)
            if w.ndim!=2:raise ValueError(f'{name}: expected matrix, got {w.shape}')
            if w.shape[1]==x.shape[-1]: y=x @ w.T
            elif w.shape[0]==x.shape[-1]: y=x @ w
            else: raise ValueError(f'{name}: matrix shape {w.shape} incompatible with {x.shape}')
        # Qwen2 uses biased Q/K/V projections while Llama commonly does not.
        # Honor an explicitly stored bias tensor without assuming it exists.
        bias_name=name[:-6]+'bias' if name.endswith('.weight') else ''
        if bias_name and bias_name in self.gguf.tensors:
            bias=self.w(bias_name)
            if bias.shape[-1] != y.shape[-1]:
                raise ValueError(f'{bias_name}: bias shape {bias.shape} incompatible with output {y.shape}')
            y=y+bias
        return y
    @staticmethod
    def norm(x,w,eps):return x*(1.0/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps))*w
    def _rope_sequence(self, a, positions):
        """Apply the architecture's RoPE layout to [tokens, heads, head_dim].

        GGUF/ggml uses adjacent-pair (NORM) RoPE for Llama weights and
        split-half (NEOX) RoPE for Qwen2/Qwen3. Applying NEOX to Llama silently
        produces finite but incorrect logits, which is especially visible on
        instruction-tuned models such as Saiga.
        """
        rotary = min(self.rope_dim, self.head_dim)
        rotary -= rotary % 2
        half = rotary // 2
        inv = 1.0 / (self.rope_base ** (np.arange(half, dtype=np.float32) / max(half, 1)))
        angle = np.asarray(positions, dtype=np.float32)[:, None] * inv[None, :]
        c, s = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]
        out = a.copy()
        if self.arch == 'llama':
            pairs = a[..., :rotary].reshape(*a.shape[:-1], half, 2)
            even = pairs[..., 0].copy()
            odd = pairs[..., 1].copy()
            out_pairs = out[..., :rotary].reshape(*a.shape[:-1], half, 2)
            out_pairs[..., 0] = even * c - odd * s
            out_pairs[..., 1] = even * s + odd * c
        else:
            left = a[..., :half].copy()
            right = a[..., half:rotary].copy()
            out[..., :half] = left * c - right * s
            out[..., half:rotary] = left * s + right * c
        return out

    def forward_sequence(self, tokens, stop_event=None):
        """Batched causal prompt prefill; return logits for the last prompt token."""
        token_ids = list(tokens)
        if not token_ids:
            raise ValueError('forward_sequence requires at least one token')
        if len(token_ids) > self.cache.max_seq:
            token_ids = token_ids[-self.cache.max_seq:]
        positions = np.arange(len(token_ids), dtype=np.int64)
        emb = self.gguf.find('token_embd.weight', 'tok_embeddings.weight')
        if hasattr(self.gguf, 'tensor_row'):
            x = np.stack([self.gguf.tensor_row(emb, int(token)) for token in token_ids]).astype(np.float32, copy=False)
        else:
            x = np.stack([self.w(emb)[int(token)] for token in token_ids]).astype(np.float32, copy=False)
        token_count = len(token_ids)
        for layer in range(self.n_layers):
            if stop_event is not None and stop_event.is_set():
                return None
            p = f'blk.{layer}.'
            residual = x
            h = self.norm(x, self.w(self.find(layer, 'attn_norm.weight')), self.eps)
            q = self.mat(h, self.find(layer, 'attn_q.weight')).reshape(token_count, self.n_heads, self.head_dim)
            k = self.mat(h, self.find(layer, 'attn_k.weight')).reshape(token_count, self.n_kv_heads, self.head_dim)
            v = self.mat(h, self.find(layer, 'attn_v.weight')).reshape(token_count, self.n_kv_heads, self.head_dim)
            q_norm_name = f'blk.{layer}.attn_q_norm.weight'
            k_norm_name = f'blk.{layer}.attn_k_norm.weight'
            if q_norm_name in self.gguf.tensors:
                q = self.norm(q, self.w(q_norm_name), self.eps)
            if k_norm_name in self.gguf.tensors:
                k = self.norm(k, self.w(k_norm_name), self.eps)
            q = self._rope_sequence(q, positions)
            k = self._rope_sequence(k, positions)
            self.cache.k[layer, positions] = k
            self.cache.v[layer, positions] = v
            group = self.n_heads // self.n_kv_heads
            q_grouped = q.reshape(token_count, self.n_kv_heads, group, self.head_dim)
            keys = self.cache.k[layer, :token_count]
            values = self.cache.v[layer, :token_count]
            scores = np.einsum('thgd,shd->thgs', q_grouped, keys, optimize=True) / math.sqrt(self.head_dim)
            future = positions[None, :] > positions[:, None]
            if self.sliding_window > 0:
                too_old = positions[None, :] < (positions[:, None] - self.sliding_window + 1)
                future |= too_old
            scores = np.where(future[:, None, None, :], -np.inf, scores)
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            att = np.einsum('thgs,shd->thgd', probs, values, optimize=True).reshape(token_count, self.dim)
            x = residual + self.mat(att, self.find(layer, 'attn_output.weight'))
            residual = x
            ffn_norm_name = 'post_attention_norm.weight' if self.arch == 'qwen3' else 'ffn_norm.weight'
            h = self.norm(x, self.w(self.find(layer, ffn_norm_name)), self.eps)
            gate = self.mat(h, self.find(layer, 'ffn_gate.weight'))
            up = self.mat(h, self.find(layer, 'ffn_up.weight'))
            activated = gate / (1.0 + np.exp(-np.clip(gate, -80, 80)))
            x = residual + self.mat(activated * up, self.find(layer, 'ffn_down.weight'))
        self.cache.length = token_count
        out = self.norm(x[-1], self.w(self.gguf.find('output_norm.weight', 'norm.weight')), self.eps)
        outname = self.gguf.find('output.weight')
        if outname:
            return self.mat(out, outname)
        return self.gguf.matvec(emb, out) if hasattr(self.gguf, 'matvec') else out @ self.w(emb).T

    def forward(self,token,position):
        m=self.gguf.metadata; arch=self.arch
        emb=self.gguf.find('token_embd.weight','tok_embeddings.weight')
        if hasattr(self.gguf, 'tensor_row') and emb in self.gguf.tensors:
            x=self.gguf.tensor_row(emb, int(token)).astype(np.float32,copy=False)
        else:
            x=self.w(emb)[int(token)].astype(np.float32,copy=True)
        for layer in range(self.n_layers):
            pre=x.copy(); pfx=f'blk.{layer}.'
            def get(s):return self.w(self.find(layer,s))
            an=get('attn_norm.weight'); x=self.norm(x,an,self.eps)
            q=self.mat(x,self.find(layer,'attn_q.weight')).reshape(self.n_heads,self.head_dim)
            k=self.mat(x,self.find(layer,'attn_k.weight')).reshape(self.n_kv_heads,self.head_dim)
            v=self.mat(x,self.find(layer,'attn_v.weight')).reshape(self.n_kv_heads,self.head_dim)
            # Qwen3 adds per-head Q/K RMSNorm; older layouts omit these tensors.
            q_norm_name=f'blk.{layer}.attn_q_norm.weight'
            k_norm_name=f'blk.{layer}.attn_k_norm.weight'
            if q_norm_name in self.gguf.tensors:q=self.norm(q,self.w(q_norm_name),self.eps)
            if k_norm_name in self.gguf.tensors:k=self.norm(k,self.w(k_norm_name),self.eps)
            # rope.dimension_count can be smaller than head_dim; dimensions
            # outside that rotary prefix must remain unchanged. Llama GGUF uses
            # adjacent-pair RoPE, while Qwen2/Qwen3 use the NeoX split-half form.
            rotary=min(self.rope_dim,self.head_dim); rotary-=rotary%2; half=rotary//2
            inv=1.0/(self.rope_base**(np.arange(half,dtype=np.float32)/max(half,1)))
            if self.rope_scaling_type == 'llama3' and self.rope_scaling_factor > 1.0:
                # Llama 3.1/3.2 frequency interpolation: preserve high-frequency
                # rotations and scale low-frequency rotations for long context.
                wavelength=(2.0*np.pi)/np.maximum(inv,1e-12)
                low_wavelength=self.rope_original_context/max(self.rope_low_freq,1e-6)
                high_wavelength=self.rope_original_context/max(self.rope_high_freq,1e-6)
                smooth=(self.rope_original_context/np.maximum(wavelength,1e-12)-self.rope_low_freq)/max(self.rope_high_freq-self.rope_low_freq,1e-6)
                interpolated=(1.0-smooth)*(inv/self.rope_scaling_factor)+smooth*inv
                inv=np.where(wavelength>low_wavelength,inv/self.rope_scaling_factor,
                    np.where(wavelength<high_wavelength,inv,interpolated))
            angle=position*inv
            if self.rope_scaling_type == 'linear' and self.rope_scaling_factor > 1.0:
                angle=angle/self.rope_scaling_factor
            c=np.cos(angle); s=np.sin(angle)
            def rope(a):
                z=a.copy()
                if arch == 'llama':
                    pairs=a[...,:rotary].reshape(*a.shape[:-1],half,2)
                    even=pairs[...,0].copy(); odd=pairs[...,1].copy()
                    zp=z[...,:rotary].reshape(*a.shape[:-1],half,2)
                    zp[...,0]=even*c-odd*s; zp[...,1]=even*s+odd*c
                else:
                    u=a[...,:half].copy(); vv=a[...,half:rotary].copy(); z[...,:half]=u*c-vv*s; z[...,half:rotary]=u*s+vv*c
                return z
            q=rope(q); k=rope(k); self.cache.k[layer,position]=k; self.cache.v[layer,position]=v
            cache_start=max(0,position+1-self.sliding_window) if self.sliding_window>0 else 0
            ks=self.cache.k[layer,cache_start:position+1]; vs=self.cache.v[layer,cache_start:position+1]
            groups=self.n_heads//self.n_kv_heads; kh=np.repeat(ks,groups,axis=1).transpose(1,0,2); vh=np.repeat(vs,groups,axis=1).transpose(1,0,2)
            scores=np.einsum('hd,hld->hl',q,kh,optimize=True)/math.sqrt(self.head_dim); scores-=scores.max(axis=-1,keepdims=True); probs=np.exp(scores); probs/=probs.sum(axis=-1,keepdims=True)
            att=np.einsum('hl,hld->hd',probs,vh,optimize=True).reshape(self.dim)
            x=pre+self.mat(att,self.find(layer,'attn_output.weight'))
            pre=x.copy()
            ffn_norm_name='post_attention_norm.weight' if arch == 'qwen3' else 'ffn_norm.weight'
            x=self.norm(x,get(ffn_norm_name),self.eps)
            gate=self.mat(x,self.find(layer,'ffn_gate.weight')); up=self.mat(x,self.find(layer,'ffn_up.weight')); sw=gate/(1.0+np.exp(-np.clip(gate,-80,80)))
            x=pre+self.mat(sw*up,self.find(layer,'ffn_down.weight'))
        out=self.norm(x,self.w(self.gguf.find('output_norm.weight','norm.weight')),self.eps)
        outname=self.gguf.find('output.weight')
        if outname:
            return self.mat(out, outname)
        if hasattr(self.gguf, 'matvec') and emb in self.gguf.tensors:
            return self.gguf.matvec(emb, out)
        return out @ self.w(emb).T
    def generate(self,prompt_ids,tokenizer,stop_event,emit,temperature=.8,top_p=.95,repetition_penalty=1.1,max_new=256,top_k=0):
        from .sampler import sample
        self.cache.reset(); ids=list(prompt_ids)
        self.last_prefill_seconds = 0.0
        self.last_generated_tokens = 0
        if not ids:ids=[tokenizer.bos]
        logits=None
        prompt=ids[-self.cache.max_seq:]
        prefill_started = time.perf_counter()
        if stop_event.is_set(): return
        # Architectures with recurrent state may provide their own prefill path
        # (for example ZAYA's CCA state and reusable prompt-prefix snapshots).
        if hasattr(self, 'prefill_prompt'):
            logits = self.prefill_prompt(prompt, stop_event, tokenizer)
            if logits is None or stop_event.is_set(): return
        # This batched implementation currently models the standard Llama
        # attention block only. Qwen2/Qwen3 have architecture-specific details
        # (including rotary scaling), so keep their proven tokenwise prefill
        # until a separately verified batched implementation exists.
        elif (self.arch == 'llama' and self.rope_scaling_type in {'none', ''}
                and self.rope_scaling_factor <= 1.0 and hasattr(self, 'forward_sequence')):
            logits = self.forward_sequence(prompt, stop_event)
            if logits is None or stop_event.is_set(): return
            self.cache.length = len(prompt)
        else:
            for pos,tok in enumerate(prompt):
                if stop_event.is_set():return
                logits=self.forward(tok,pos)
                self.cache.length=pos+1
        self.last_prefill_seconds = time.perf_counter() - prefill_started
        self.decode_profiler.start_decode()
        decoder=tokenizer.new_decoder()
        def flush_decoder():
            tail=decoder.decode(b'',final=True)
            decoder.reset()
            if tail:emit(tail)
        for generated_idx in range(max_new):
            if stop_event.is_set() or self.cache.length>=self.cache.max_seq:
                flush_decoder(); return
            sample_started = time.perf_counter() if self.decode_profiler.enabled else 0.0
            nxt=sample(logits,ids,temperature,top_p,repetition_penalty,top_k=top_k,
                       penalty_exempt=getattr(tokenizer,'stop_ids',{tokenizer.eos}))
            if self.decode_profiler.enabled:
                self.decode_profiler.record_sampling(time.perf_counter() - sample_started)
            if nxt in getattr(tokenizer,'stop_ids',{tokenizer.eos}):
                flush_decoder(); return
            ids.append(nxt)
            self.last_generated_tokens += 1
            emit(tokenizer.decode_token(nxt,decoder))
            # The last requested token needs no follow-up logits pass. Also
            # honor Stop immediately rather than waiting for another matvec.
            if generated_idx + 1 >= max_new or stop_event.is_set():
                flush_decoder(); return
            forward_started = time.perf_counter() if self.decode_profiler.enabled else 0.0
            logits=self.forward(nxt,self.cache.length)
            if self.decode_profiler.enabled:
                self.decode_profiler.record_forward(time.perf_counter() - forward_started)
            self.cache.length+=1
        flush_decoder()


class Qwen2Model(LlamaModel):
    """Qwen2/Qwen2.5 decoder using the shared attention/SwiGLU primitives.

    This is an explicit architecture implementation, not a fallback for other
    Qwen generations. Qwen3/Qwen3.5 have different block details and are not
    routed here.
    """

    def __init__(self, gguf, max_seq=2048):
        architecture = str(gguf.metadata.get('general.architecture', 'qwen2')).lower()
        if architecture != 'qwen2':
            raise ValueError(f'Qwen2Model requires general.architecture=qwen2, got {architecture!r}')
        super().__init__(gguf, max_seq=max_seq)


class Qwen3Model(LlamaModel):
    """Qwen3 dense decoder with per-head Q/K RMSNorm."""

    def __init__(self, gguf, max_seq=2048):
        architecture = str(gguf.metadata.get('general.architecture', 'qwen3')).lower()
        if architecture != 'qwen3':
            raise ValueError(f'Qwen3Model requires general.architecture=qwen3, got {architecture!r}')
        super().__init__(gguf, max_seq=max_seq)


class Qwen3Model(LlamaModel):
    """Qwen3 decoder: shared dense attention/MLP core plus per-head Q/K norms."""

    def __init__(self, gguf, max_seq=2048):
        architecture = str(gguf.metadata.get('general.architecture', 'qwen3')).lower()
        if architecture != 'qwen3':
            raise ValueError(f'Qwen3Model requires general.architecture=qwen3, got {architecture!r}')
        super().__init__(gguf, max_seq=max_seq)
