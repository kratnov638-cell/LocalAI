"""Minimal GGUF v2/v3 reader with mmap-backed tensor views."""
from __future__ import annotations
import mmap, struct, os
from dataclasses import dataclass
from pathlib import Path
import numpy as np

MAGIC=b'GGUF'
# GGUF value types
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)
_STRUCT={UINT8:'B',INT8:'b',UINT16:'H',INT16:'h',UINT32:'I',INT32:'i',FLOAT32:'f',BOOL:'?',UINT64:'Q',INT64:'q',FLOAT64:'d'}

def _native_thread_count(rows, ggml_type, *, batched=False):
    # AVX2 kernels began accumulating SIMD lanes across quantization blocks,
    # Decode GEMV and prompt-prefill GEMM have different scaling. Measurements
    # across the supported models favor 8 workers for medium/large Q4_K and
    # large vocabulary heads, 7 for medium Q6_K, and 6 for Q8_0. Keep the
    # batched prefill policy separate because short batches scale differently.
    # Matrices below 256 rows are serialized by the native kernel regardless.
    # Optional decode-only override for reproducible hardware tuning. Keep
    # prompt-prefill policy independent because short batched GEMMs scale
    # differently from one-token decode GEMVs.
    if not batched:
        override = os.environ.get('LOCALAI_NATIVE_DECODE_THREADS', '').strip()
        if override:
            try:
                requested = int(override)
            except ValueError:
                requested = 0
            if requested > 0:
                return min(requested, os.cpu_count() or 1)
    if rows >= 65536 and ggml_type != 8:
        requested = 8
    elif not batched and ggml_type == 12 and rows >= 4096:
        requested = 8
    elif not batched and ggml_type == 14 and rows >= 4096:
        requested = 7
    elif ggml_type == 8 or ggml_type in (12, 13, 14):
        requested = 6
    else:
        requested = 4
    return min(requested, os.cpu_count() or 1)

@dataclass
class TensorInfo:
    name:str; shape:tuple[int,...]; ggml_type:int; offset:int; nbytes:int=0
class GGUFFile:
    def __init__(self,path):
        self.path=Path(path)
        self._file=self.path.open('rb')
        self._map=None
        try:
            self._map=mmap.mmap(self._file.fileno(),0,access=mmap.ACCESS_READ)
            self.metadata={}; self.tensors={}; self.version=0; self.alignment=32
            # Optional decoded-matrix cache stores float16 copies. It is bounded
            # and non-evicting so the first layers remain hot across token steps.
            # Keep the huge vocabulary head in its own budget so layer weights
            # cannot evict it on every decode step.
            self._matrix_cache = {}
            self._matrix_cache_bytes = 0
        # These bounded caches are a fallback for quant types without a direct
        # native kernel. Q4_K/Q6_K should stay packed when the project-owned
        # AVX2 kernels are available; expanding them to FP32 increases memory
        # traffic and makes the first token much slower.
            self._matrix_cache_limit = max(0, int(float(os.environ.get('LOCALAI_DEQUANT_CACHE_GB', '4')) * (1024 ** 3)))
            cache_dtype_name = os.environ.get('LOCALAI_DEQUANT_CACHE_DTYPE', 'float32').strip().lower()
            if cache_dtype_name not in {'float16', 'float32'}:
                raise ValueError('LOCALAI_DEQUANT_CACHE_DTYPE must be float16 or float32')
            self._matrix_cache_dtype = np.float32 if cache_dtype_name == 'float32' else np.float16
            self._head_cache = {}
            self._head_cache_bytes = 0
            # The vocabulary projection is read on every generated token. Keep
            # it in FP32 so matvec can avoid allocating/converting a 2 GiB
            # FP16 matrix on every step; this costs ~3.8 GiB for a 248k x 4096
            # head and is separate from the FP16 layer cache.
            self._head_cache_limit = max(0, int(float(os.environ.get('LOCALAI_HEAD_CACHE_GB', '4.5')) * (1024 ** 3)))
            self._parse()
        except Exception:
            if self._map is not None:
                self._map.close()
            self._file.close()
            raise
    def close(self):
        self._matrix_cache.clear(); self._head_cache.clear()
        self._matrix_cache_bytes = self._head_cache_bytes = 0
        self._map.close(); self._file.close()
    def _read(self,fmt):
        n=struct.calcsize('<'+fmt); p=self._map.tell() if hasattr(self._map,'tell') else 0
        b=self._map.read(n)
        if len(b)!=n: raise ValueError(f'{self.path}: truncated GGUF at byte {p}')
        return struct.unpack('<'+fmt,b)[0]
    def _str(self):
        n=self._read('Q')
        if n>self._map.size(): raise ValueError('Invalid GGUF string length')
        b=self._map.read(n)
        if len(b)!=n: raise ValueError('Truncated GGUF string')
        return b.decode('utf-8',errors='replace')
    def _value(self,t):
        if t==STRING:return self._str()
        if t==ARRAY:
            et=self._read('I'); n=self._read('Q')
            if n>100_000_000: raise ValueError('Unreasonable GGUF array length')
            return [self._value(et) for _ in range(n)]
        if t not in _STRUCT: raise ValueError(f'Unsupported GGUF metadata value type {t}')
        return self._read(_STRUCT[t])
    def _parse(self):
        if self._map.read(4)!=MAGIC: raise ValueError(f'{self.path} is not a GGUF file')
        self.version=self._read('I')
        if self.version not in (2,3): raise ValueError(f'GGUF version {self.version} unsupported; expected v2/v3')
        nt=self._read('Q'); nk=self._read('Q')
        if nt>10_000_000 or nk>1_000_000: raise ValueError('Invalid GGUF counts')
        for _ in range(nk):
            k=self._str(); typ=self._read('I'); self.metadata[k]=self._value(typ)
        self.alignment=int(self.metadata.get('general.alignment',32))
        if self.alignment<=0 or self.alignment>4096: self.alignment=32
        infos=[]
        names=set()
        for _ in range(nt):
            name=self._str(); nd=self._read('I')
            if nd<1 or nd>8: raise ValueError(f'Tensor {name}: invalid dimension count {nd}')
            dims=tuple(self._read('Q') for _ in range(nd)); typ=self._read('I'); off=self._read('Q')
            if name in names: raise ValueError(f'Duplicate tensor name in GGUF: {name!r}')
            if any(dim == 0 for dim in dims): raise ValueError(f'Tensor {name}: dimensions must be positive')
            if off % self.alignment: raise ValueError(f'Tensor {name}: data offset {off} is not aligned to {self.alignment} bytes')
            names.add(name)
            infos.append(TensorInfo(name,tuple(reversed(dims)),typ,off))
        pos=self._map.tell(); data_start=(pos+self.alignment-1)//self.alignment*self.alignment
        ranges=[]
        for x in infos:
            x.offset += data_start
            # Validate all tensor ranges whose GGML block layout is known. An
            # unknown type remains inspectable, but fails explicitly on access.
            try:
                size=self.tensor_bytes(x)
            except NotImplementedError:
                size=None
            if size is not None:
                end=x.offset+size
                if end>self._map.size():
                    raise ValueError(f'Tensor {x.name} extends past end of GGUF (end={end}, file_size={self._map.size()})')
                ranges.append((x.offset,end,x.name))
            self.tensors[x.name]=x
        ranges.sort()
        for previous,current in zip(ranges,ranges[1:]):
            if current[0] < previous[1]:
                raise ValueError(f'Tensor data ranges overlap: {previous[2]!r} and {current[2]!r}')
    def tensor_bytes(self,info):
        from .dequant import block_layout
        block,bytes_per=block_layout(info.ggml_type)
        n=int(np.prod(info.shape,dtype=np.int64))
        if n%block: raise ValueError(f'Tensor {info.name} has {n} values, not divisible by block size {block}')
        return (n//block)*bytes_per
    def raw_tensor(self,name):
        x=self.tensors[name]; n=self.tensor_bytes(x); end=x.offset+n
        if end>self._map.size(): raise ValueError(f'Tensor {name} extends past end of GGUF')
        return memoryview(self._map)[x.offset:end]
    def tensor(self,name):
        from .dequant import dequantize
        x=self.tensors[name]
        try:return dequantize(self.raw_tensor(name),x.ggml_type,x.shape)
        except Exception as e: raise type(e)(f'tensor "{name}" (GGML type {x.ggml_type}): {e}') from e

    def tensor_row(self, name, row):
        """Decode one row of a 2-D tensor without materializing the full matrix."""
        from .dequant import block_layout, dequantize
        info = self.tensors[name]
        if len(info.shape) != 2:
            raise ValueError(f'{name}: tensor_row requires a 2-D tensor, got {info.shape}')
        rows, cols = info.shape
        if not 0 <= int(row) < rows:
            raise IndexError(f'{name}: row {row} outside [0, {rows})')
        block, bytes_per_block = block_layout(info.ggml_type)
        if cols % block:
            raise ValueError(f'{name}: row length {cols} is not divisible by quantization block {block}')
        row_bytes = (cols // block) * bytes_per_block
        start = info.offset + int(row) * row_bytes
        raw = memoryview(self._map)[start:start + row_bytes]
        return dequantize(raw, info.ggml_type, (cols,))

    def matvec(self, name, vector, *, row_chunk=1024):
        """Compute W @ x from a GGUF matrix [out, in] with bounded RAM.

        This uses our NumPy quantization decoders, not an external inference
        runtime. It avoids expanding every 9B checkpoint weight to float32.
        """
        from .dequant import block_layout, dequantize
        info = self.tensors[name]
        if len(info.shape) != 2:
            raise ValueError(f'{name}: matvec requires a 2-D tensor, got {info.shape}')
        rows, cols = info.shape
        x = np.asarray(vector, dtype=np.float32)
        if x.ndim != 1 or x.size != cols:
            raise ValueError(f'{name}: expected vector shape ({cols},), got {x.shape}')
        if row_chunk < 1:
            raise ValueError('row_chunk must be positive')
        block, bytes_per_block = block_layout(info.ggml_type)
        if cols % block:
            raise ValueError(f'{name}: row length {cols} is not divisible by quantization block {block}')
        row_bytes = (cols // block) * bytes_per_block
        out = np.empty(rows, dtype=np.float32)
        # Prefer packed AVX2 dot products before considering decoded-weight
        # caching. Expanding Q8_0/Q4_K/Q5_K/Q6_K weights to FP32 adds memory
        # traffic and can spend seconds materializing matrices the SIMD kernels
        # consume directly. Keep bounded caches as a fallback when native code
        # is unavailable or for quant types without a direct kernel.
        if info.ggml_type in (2, 8, 12, 13, 14):
            try:
                from .native_quant import matvec as native_matvec
                native_result = native_matvec(
                    self.raw_tensor(name), x, rows, cols, info.ggml_type,
                    threads=_native_thread_count(rows, info.ggml_type),
                )
                if native_result is not None:
                    return native_result
            except Exception:
                pass
        cached = self._head_cache.get(name)
        if cached is None:
            cached = self._matrix_cache.get(name)
        if cached is not None:
            # Cached half-precision matrices are already decoded, so use much
            # larger chunks to amortize Python/NumPy dispatch overhead. 2048
            # rows at 4096 columns needs only ~32 MiB for the float32 tile.
            cached_chunk = max(row_chunk, 2048)
            for first in range(0, rows, cached_chunk):
                last = min(first + cached_chunk, rows)
                weights = cached[first:last].astype(np.float32, copy=False)
                out[first:last] = weights @ x
            return out

        is_head = name in {'output.weight', 'token_embd.weight'}
        cache_dtype = np.float32 if is_head else self._matrix_cache_dtype
        cache_bytes = rows * cols * np.dtype(cache_dtype).itemsize
        # Cache only supported quantized/half matrices; the configurable cache
        # dtype is independent of the on-disk quantization format.
        can_cache = info.ggml_type in (1, 2, 3, 6, 7, 8, 12, 13, 14)
        cache_target = None
        if can_cache and is_head and cache_bytes <= self._head_cache_limit - self._head_cache_bytes:
            cache_target = np.empty((rows, cols), dtype=cache_dtype)
        elif can_cache and cache_bytes <= self._matrix_cache_limit - self._matrix_cache_bytes:
            cache_target = np.empty((rows, cols), dtype=cache_dtype)
        # If the bounded FP32 cache cannot hold this whole matrix, consume
        # Q4_K/Q6_K blocks directly. Re-dequantizing a multi-gigabyte matrix
        # on every generated token is much more expensive than decoding each
        # quantized value only for its dot product.
        if cache_target is None and info.ggml_type in (12, 14):
            raw = memoryview(self._map)[info.offset:info.offset + rows * row_bytes]
            # Prefer project-owned AVX2 SIMD kernels; retain Numba as a portable
            # fallback for systems without the MSVC toolchain/AVX2 runtime.
            try:
                from .native_quant import matvec as native_matvec
                native_result = native_matvec(raw, x, rows, cols, info.ggml_type,
                                              threads=_native_thread_count(rows, info.ggml_type))
                if native_result is not None:
                    return native_result
            except Exception:
                pass
            try:
                from .numba_kernels import matvec_q4_k_direct, matvec_q6_k_direct
            except ImportError:
                pass
            else:
                raw = memoryview(self._map)[info.offset:info.offset + rows * row_bytes]
                packed = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols // block, bytes_per_block)
                blocks_per_row = cols // block
                if info.ggml_type == 12:
                    params = np.frombuffer(raw, dtype='<f2').reshape(rows, blocks_per_row, 72)
                    return matvec_q4_k_direct(packed, params[:, :, 0].astype(np.float32),
                                              params[:, :, 1].astype(np.float32), x)
                params = np.frombuffer(raw, dtype='<f2').reshape(rows, blocks_per_row, 105)
                return matvec_q6_k_direct(packed, params[:, :, 104].astype(np.float32), x)

        for first in range(0, rows, row_chunk):
            count = min(row_chunk, rows - first)
            start = info.offset + first * row_bytes
            raw = memoryview(self._map)[start:start + count * row_bytes]
            weights = dequantize(raw, info.ggml_type, (count, cols))
            out[first:first + count] = weights @ x
            if cache_target is not None:
                cache_target[first:first + count] = weights
        if cache_target is not None:
            if is_head:
                self._head_cache[name] = cache_target
                self._head_cache_bytes += cache_bytes
            else:
                self._matrix_cache[name] = cache_target
                self._matrix_cache_bytes += cache_bytes
        return out

    def matmul(self, name, vectors, *, row_chunk=1024):
        """Compute a batch of row vectors against GGUF W[out, in].

        Unlike repeatedly calling matvec, this decodes each matrix tile once
        for the whole prompt, which is important for CPU prefill of long prompts.
        """
        from .dequant import block_layout, dequantize
        info = self.tensors[name]
        if len(info.shape) != 2:
            raise ValueError(f'{name}: matmul requires a 2-D tensor, got {info.shape}')
        rows, cols = info.shape
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != cols:
            raise ValueError(f'{name}: expected batch shape (tokens, {cols}), got {x.shape}')
        if row_chunk < 1:
            raise ValueError('row_chunk must be positive')
        block, bytes_per_block = block_layout(info.ggml_type)
        if cols % block:
            raise ValueError(f'{name}: row length {cols} is not divisible by quantization block {block}')
        row_bytes = (cols // block) * bytes_per_block
        out = np.empty((x.shape[0], rows), dtype=np.float32)
        if info.ggml_type in (2, 8, 12, 13, 14):
            # Prefer the project-owned packed SIMD batch kernel. It avoids
            # allocating large FP32 tiles during prefill; if native AVX2 is not
            # available, bounded dequantization + BLAS remains the fallback.
            try:
                from .native_quant import matmul as native_matmul
                native_result = native_matmul(
                    self.raw_tensor(name), x, rows, cols, info.ggml_type,
                    threads=_native_thread_count(rows, info.ggml_type, batched=True),
                )
                if native_result is not None:
                    return native_result
            except Exception:
                pass
            for first in range(0, rows, row_chunk):
                count = min(row_chunk, rows - first)
                start = info.offset + first * row_bytes
                raw = memoryview(self._map)[start:start + count * row_bytes]
                weights = dequantize(raw, info.ggml_type, (count, cols))
                out[:, first:first + count] = x @ weights.T
            return out
        cached = self._head_cache.get(name)
        if cached is None:
            cached = self._matrix_cache.get(name)
        if cached is not None:
            cached_chunk = max(row_chunk, 2048)
            for first in range(0, rows, cached_chunk):
                last = min(first + cached_chunk, rows)
                weights = cached[first:last].astype(np.float32, copy=False)
                out[:, first:last] = x @ weights.T
            return out

        is_head = name in {'output.weight', 'token_embd.weight'}
        cache_dtype = np.float32 if is_head else self._matrix_cache_dtype
        cache_bytes = rows * cols * np.dtype(cache_dtype).itemsize
        can_cache = info.ggml_type in (1, 2, 3, 6, 7, 8, 12, 13, 14)
        cache_target = None
        if can_cache and is_head and cache_bytes <= self._head_cache_limit - self._head_cache_bytes:
            cache_target = np.empty((rows, cols), dtype=cache_dtype)
        elif can_cache and cache_bytes <= self._matrix_cache_limit - self._matrix_cache_bytes:
            cache_target = np.empty((rows, cols), dtype=cache_dtype)
        for first in range(0, rows, row_chunk):
            count = min(row_chunk, rows - first)
            start = info.offset + first * row_bytes
            raw = memoryview(self._map)[start:start + count * row_bytes]
            weights = dequantize(raw, info.ggml_type, (count, cols))
            out[:, first:first + count] = x @ weights.T
            if cache_target is not None:
                cache_target[first:first + count] = weights
        if cache_target is not None:
            if is_head:
                self._head_cache[name] = cache_target
                self._head_cache_bytes += cache_bytes
            else:
                self._matrix_cache[name] = cache_target
                self._matrix_cache_bytes += cache_bytes
        return out
    def find(self,*suffixes):
        # Prefer exact names before suffix matches. Without this, asking for
        # `output.weight` can accidentally return `blk.N.attn_output.weight`.
        for s in suffixes:
            if s in self.tensors:return s
        for s in suffixes:
            for name in self.tensors:
                if name.endswith('.'+s): return name
        return None
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()






