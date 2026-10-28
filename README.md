# LocalAI - custom NumPy GGUF inference engine

LocalAI is a desktop chat application with a from-scratch Python/NumPy inference implementation and project-owned native C++ AVX2/FMA kernels for hot quantized operations. It does not wrap llama.cpp, ggml, transformers, PyTorch, ONNX Runtime, or another prebuilt inference engine.

## Architecture strategy

The engine is deliberately split into shared format/runtime infrastructure and model-family implementations:

- **Shared:** GGUF v2/v3 parsing, mmap-backed tensor access, quantization decoders, tokenizer infrastructure, sampling, chat UI and session lifecycle.
- **Architecture registry:** `engine/architectures.py` dispatches the GGUF `general.architecture` value to a specific forward-pass implementation. Unknown architectures fail explicitly; they are never guessed to be Llama.
- **Current forward-pass implementations:** `llama`, `mistral`, `qwen2`, `qwen3`, and a text-only `qwen35` implementation are explicitly registered. The current target set is `model-q4_K.gguf` (Saiga Llama 3 Q4_K), `model-q8_0.gguf` (Saiga Llama 3 Q8_0), `Qwen3.5-9B-Q4_K_M.gguf`, and `Qwen3.5-9B-abliterated-vision-Q4_K_M.gguf`. Models are downloaded by the user and selected as local files; LocalAI does not download model weights. Registration and unit tests do not prove every checkpoint is numerically validated.
- **Qwen3.5 vision:** GGUF inspection detects the vision tower, but image/video input and multimodal position IDs are not implemented. Text-only prompts are the target for this decoder.
- **Quantized RAM use:** real GGUF matrices are decoded in bounded row chunks, with configurable fallback caches (`LOCALAI_DEQUANT_CACHE_GB`, default 4 GiB FP32 for layer matrices; `LOCALAI_HEAD_CACHE_GB`, default 4.5 GiB FP32 for the frequently reused output head). When the project-owned AVX2 kernel is available, Q4_0/Q8_0/Q4_K/Q5_K/Q6_K matrices stay packed and bypass FP32 caching to avoid costly expansion and excess memory traffic. Q4_0 uses 16-bit quantized activations per 32-value group; Q4_K/Q5_K use 8-bit activations per 32-value group for direct AVX2 byte-dot products; Q6_K uses 16-bit activations per 16-value group. Other quant types use the bounded cache and portable decoder paths. `LOCALAI_DEQUANT_CACHE_DTYPE=float16` reduces fallback cache RAM but adds conversion work during each token.
- **Native worker policy:** decode GEMV uses eight logical CPU threads for medium/large Q4_K matrices and seven for medium Q6_K matrices on the target i3-12100F; batched prompt-prefill uses the more conservative six-thread policy for K-quants, while large vocabulary projections use eight. Q8_0 uses six. These are measured heuristics, not guarantees that more threads are always faster.
- **Prefill reuse:** the native batched-matmul kernel decodes each quantized weight row once and reuses it across up to eight prompt positions before advancing to the next row.
- **Saiga sampling:** when `general.name` identifies `saiga_llama3_8b`, untouched UI defaults use temperature 0.2, top-p 0.9, top-k 30 and repetition penalty 1.12. User-edited temperature/top-p values are preserved. No default system prompt is injected because a real A/B run on the installed GGUFs made the output worse.
- **Adding model families:** use `engine/architectures.py` for explicit dispatch and a dedicated forward-pass class for each distinct block design. Unknown architectures still fail closed; do not route them to Llama by default.

## Initial real-model support set

The current target set is selected from GGUF metadata and tensor types, not by hardcoded filenames:

- `model-q4_K.gguf` and `model-q8_0.gguf` - Llama 3 / Saiga Llama 3 8B checkpoints.
- `Qwen3.5-9B-Q4_K_M.gguf` - standard Qwen3.5 9B text generation.
- `Qwen3.5-9B-abliterated-vision-Q4_K_M.gguf` - text generation from the vision-tagged Qwen3.5 checkpoint.

GGUF files are read-only inputs: LocalAI does not rewrite, requantize, or patch model weights. Qwen3.5 image/video input is not implemented yet; the vision-tagged checkpoint currently runs text-only prompts.

## Run on Windows

1. Install 64-bit Python 3.10+.
2. Put a supported `.gguf` file in `models/` or select it in the app.
3. Run `run.bat`.

Linux/macOS: create a virtual environment, install `requirements.txt`, then run `python main.py`.

## Implemented components

- GGUF v2/v3 header, metadata and tensor table parsing.
- Memory-mapped tensor data access.
- NumPy tensor decoders for F32/F16 and GGML Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K and Q6_K.
- Project-owned AVX2 Q4_0/Q8_0/Q4_K/Q5_K/Q6_K direct matvec and batched matmul kernels with local MSVC compilation; includes Numba-parallel block decoding and a portable NumPy fallback.
- Byte-level BPE tokenizer built from GGUF vocabulary and merge metadata, including Qwen3.5 pre-tokenizer naming.
- Llama-family operations: RMSNorm, projections, NeoX-style RoPE (including Llama 3 and linear scaling), grouped-query attention, SwiGLU, residuals and KV cache.
- Sampling, repetition penalty and streaming chat UI.
- Persistent inference sessions and an explicit architecture registry.

## Limitations

This is an educational implementation, not a drop-in replacement for mature inference runtimes. `qwen35` is text-only and still requires numerical comparison against a trusted reference. Vision, video, MoE, IQ quantization, many architecture-specific operations, and a general Jinja chat-template engine are not implemented. Batched prefill and JIT quantized decoding reduce latency, but single-token decoding is still memory-bandwidth-bound and substantially slower than mature SIMD inference engines. Passing unit tests and producing a token do not establish answer quality.

## Tests

Run from this directory:

```powershell
.\.venv\Scripts\python.exe -m compileall -q engine app main.py tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Unit tests include synthetic GGUFs; they do not replace end-to-end validation on real model files.

## Recent CPU optimization measurements

Measured on the Intel i3-12100F / 32 GB RAM using `Explain briefly why the sky is blue.`, temperature 0, top-p 1, repetition penalty 1 and 12 generated tokens. Decode tok/s excludes prompt prefill. The before numbers are one baseline run captured immediately before the kernel changes; after numbers are the median of two runs in the final kernel state. These are short same-machine smoke tests, not a direct benchmark against llama.cpp/KoboldCpp, and timing varies with CPU temperature and memory state.

| Model | Before | After | Change |
|---|---:|---:|---:|
| Saiga Llama 3 Q4_K | 3.71 tok/s | 4.82 tok/s | +30% |
| Saiga Llama 3 Q8_0 | 3.69 tok/s | 4.08 tok/s | +11% |
| Qwen3.5 9B Q4_K_M | 3.09 tok/s | 4.11 tok/s | +33% |
| Qwen3.5 9B vision-tagged Q4_K_M (text-only) | 3.22 tok/s | 4.19 tok/s | +30% |

The native AVX2 Q4_K/Q5_K/Q6_K kernels now keep scaled dot-product partials in independent SIMD-lane accumulators and horizontally reduce only after processing the row, rather than reducing every 16/32-value group to a scalar. Q4_0 uses the same vector-accumulation strategy. Q8_0 uses four explicitly unrolled 8-value accumulators across quantization blocks. The decode thread policy was retained after A/B checks because one setting did not win consistently across all four models. Benchmark completions remained coherent; Qwen3.5 greedy token choices can differ near ties because floating-point summation order changed. This verifies short English generation, not Russian instruction-following quality. The Qwen3.5 main model also completed a 32-token decode check at about 3.8 tok/s; this is still below the longer-term 6–7 tok/s goal, and further optimization remains worthwhile.

## Qwen3.5 recurrent-state optimization

The Qwen3.5 Gated DeltaNet decode step now has a Numba JIT path (`delta_net_step`) that fuses state decay with the `state @ k` contraction, then fuses the rank-one state update with the `state @ q` contraction. It removes the separate NumPy outer-product temporary and reduces four full state passes to two. The original NumPy implementation remains as a fallback when Numba is unavailable or its JIT cannot run. A numerical parity test compares both paths on randomized states.

Alternating A/B runs on the main Qwen3.5 Q4_K_M model with 48 generated tokens (same prompt and greedy settings) measured a median of **3.87 tok/s** for the NumPy recurrent step versus **4.12 tok/s** for the fused step (+6.5%). The profiler's non-matrix work fell from about 2.15 s to 1.12 s over the 48-token decode. On the vision-tagged Qwen3.5 model, two 32-token runs measured 3.94 tok/s median for NumPy and 4.28 tok/s for the fused path (+8.7%). These short measurements are subject to CPU clock and system-load variance; the recurrent-state improvement is repeatable in profiler timings, but long-context and Russian quality checks remain outstanding. Q4_K and Q8_0 do not use this Qwen-specific code path.


