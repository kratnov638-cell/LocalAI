# LocalAI — Local LLM Chat & Image Generation

**An experimental desktop application for running local GGUF language models and generating images with SDXL.** Text inference combines Python, NumPy, and project-written C++ AVX2/FMA kernels, with optional Numba acceleration for selected operations. Image generation uses a separate `stable-diffusion.cpp` runtime with CPU and optional AMD Vulkan backends.

> **Independent project notice:** this repository is a standalone experiment and is not affiliated with, derived from, or endorsed by the other project named **LocalAI** (`mudler/LocalAI`). The shared name is coincidental; the codebase, goals, and implementation are different.

## What it does

- Opens local `.gguf` model files and runs text conversations in a desktop UI.
- Reads GGUF metadata and dispatches to an explicit model-architecture implementation.
- Uses memory-mapped tensor access, quantized matrix kernels, and a portable NumPy fallback.
- Includes native AVX2/FMA kernels for selected GGML quantization formats, plus a Numba JIT path for Qwen3.5 recurrent-state updates.
- Provides temperature, top-p, maximum-new-token controls, stop generation, language selection, and a selectable chat background.
- Reports prompt-prefill and decode timing in the UI logs to help distinguish prompt processing from token generation.
- Generates SDXL images from full `.safetensors`/`.ckpt` checkpoints or a standalone SDXL U-Net `.gguf` with its companion text encoders and VAE.
- Supports AMD Vulkan acceleration for the SDXL diffusion and VAE stages when the bundled runtime exposes a Vulkan device; text encoders remain on CPU.
- Keeps model files on your machine. It does not download model weights for you.

This is an experimental inference engine, not a server, OpenAI-compatible API, or drop-in replacement for established runtimes.

## Model status

The current test set is:

| Checkpoint | Intended use | Status / caveat |
|---|---|---|
| `model-q4_K.gguf` | Saiga / Llama 3 8B, Q4_K | Text generation implemented; validate with your own prompts |
| `model-q8_0.gguf` | Saiga / Llama 3 8B, Q8_0 | Text generation implemented; validate with your own prompts |
| `Qwen3.5-9B-Q4_K_M.gguf` | Qwen3.5 9B | Text-only implementation; numerical parity still needs further validation |
| `Qwen3.5-9B-abliterated-vision-Q4_K_M.gguf` | Vision-tagged Qwen3.5 9B | Text-only; image and video input are not implemented |
| `zaya1-8b-coder-q6_k.gguf` | ZAYA1 8B Coder, Q6_K | Text generation implemented; short CPU runs reached about 6.8–7.1 decode tok/s on the test PC; Russian prose quality remains inconsistent |

Architecture dispatch currently includes Llama-style models (including the supported Mistral path), Qwen2, Qwen3, Qwen3.5, and ZAYA. Support is determined from GGUF metadata and tensor types, not just filenames. A model loading successfully or passing unit tests does not guarantee faithful output for every checkpoint.

## Run it on Windows

1. Install 64-bit Python 3.10 or newer and ensure the Python launcher (`py`) is available.
2. Clone or download this repository.
3. Download a compatible `.gguf` checkpoint separately; you can store it in the default model folder or select it from the app.
4. Run `run.bat`. The launcher creates `.venv` and installs the listed Python dependencies when needed, then starts the desktop app.

The launcher does not fetch model files. Download a compatible GGUF checkpoint separately, check its license, and select it in the app. The default model directory is `C:\LLM` when that directory exists; otherwise the app falls back to the repository's `models/` directory. Set `LOCALAI_MODELS_DIR` to use a different default directory.

To launch manually after installing dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Linux and macOS may work with a manually created virtual environment, but Windows is the primary documented path.

The application also offers a Console / Logs tab for diagnostics. Runtime logs are written under `logs/` when the configured logging path is available.

## Image generation

The **Image generation** tab uses a separate `engine/generative` package and automatically discovers `sd-cli.exe` in `engine/generative/bin`; the user does not select an external executable. The adapter invokes `stable-diffusion.cpp`, streams runtime diagnostics to the application log, supports cancellation, and validates the generated PNG (including its dimensions) before reporting success. Outputs are saved under `generated_images/`.

The image-model picker accepts `.gguf`, `.safetensors`, and `.ckpt` files. A full single-file checkpoint such as `sd_xl_base_1.0.safetensors` is passed to `stable-diffusion.cpp` through `--model` and does not trigger the separate CLIP/VAE lookup. A standalone diffusion GGUF contains only part of the pipeline, so the integration still looks for compatible CLIP-L, CLIP-G, and VAE files beside that model, under the model directory, in `models/`, or in `engine/generative/models`. Alternatively, set `LOCALAI_SDXL_CLIP_L`, `LOCALAI_SDXL_CLIP_G`, and `LOCALAI_SDXL_VAE` to explicit local file paths. For the existing `sdxl_base_1.0_Q4_K_S.gguf`, matching companion files are available from [HyperX-Sentience/SDXL-GGUF](https://huggingface.co/HyperX-Sentience/SDXL-GGUF): [CLIP-L](https://huggingface.co/HyperX-Sentience/SDXL-GGUF/resolve/main/clip/sdxl_clip_l.safetensors), [CLIP-G](https://huggingface.co/HyperX-Sentience/SDXL-GGUF/resolve/main/clip/sdxl_clip_g.safetensors), and [VAE](https://huggingface.co/HyperX-Sentience/SDXL-GGUF/resolve/main/vae/sdxl_vae.safetensors). Model weights are not downloaded automatically.

CPU remains the default. The **AMD Vulkan (GPU + CPU)** setting enables hybrid SDXL execution: the diffusion module and VAE run on the first detected Vulkan device, while text encoders run on the CPU. The current configuration limits Vulkan-managed VRAM to **3.2 GiB** and enables runtime auto-fit, allowing parameters that do not fit in VRAM to remain in system RAM. This is intended to leave headroom on a 4 GiB card such as the AMD Radeon RX 6400. The setting is enabled only when `sd-cli --list-devices` exposes a Vulkan device; the required Vulkan backend is part of the bundled runtime. CPU-only mode remains available by turning the setting off.

**Important CPU/GPU limitation:** this `stable-diffusion.cpp` revision does not support splitting the same diffusion layer graph across CPU and Vulkan. The hybrid mode assigns separate pipeline stages to the devices; it does not execute every diffusion layer simultaneously on both CPU and GPU. Actual acceleration depends on the GPU, driver, model, resolution, and available VRAM. If Vulkan is unavailable or generation fails, disable the setting to use the CPU backend.

## Tests

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m compileall -q engine app main.py tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Most unit tests use synthetic tensors or GGUF fixtures. They are useful for catching regressions but do not replace real-model quality and performance checks.

## Test environment

- **CPU:** Intel Core i3-12100F
- **RAM:** 32 GB
- **Language-model inference:** CPU
- **Image-generation GPU:** AMD Radeon RX 6400, 4 GB VRAM (Vulkan)

## Performance notes

Short CPU measurements on an Intel i3-12100F with 32 GB RAM. Decode speed excludes prompt prefill; these figures are local smoke-test results, not standardized benchmarks.

| Model | Earlier run | Optimized run | Change |
|---|---:|---:|---:|
| Saiga Llama 3 Q4_K | 3.71 tok/s | 4.82 tok/s | +30% |
| Saiga Llama 3 Q8_0 | 3.69 tok/s | 4.08 tok/s | +11% |
| Qwen3.5 9B Q4_K_M | 3.09 tok/s | 4.11 tok/s | +33% |
| Vision-tagged Qwen3.5 9B Q4_K_M, text-only | 3.22 tok/s | 4.19 tok/s | +30% |
| ZAYA1 8B Coder Q6_K | — | about 6.8–7.1 tok/s | new checkpoint / not an A/B comparison |

The Saiga and Qwen3.5 optimized figures are medians of two runs; each earlier figure is a single baseline run. ZAYA figures are approximate decode rates observed in short GUI conversations, not a controlled benchmark; prompt prefill took about 4.6–14.3 seconds in those examples. Results vary with CPU clocks, temperature, memory state, prompt length, and system load. These numbers should not be compared directly with results from other machines or runtimes.

### SDXL / Vulkan smoke tests

On the test PC above, a 512×512 SDXL generation with 12 steps completed in about **58 seconds** through the application's `GenerativeEngine` path using Vulkan for diffusion and VAE and CPU for text encoding. An earlier 12-step run took 81.5 seconds, but its thread and VAE-backend settings differed, so this is an indicative result rather than a controlled speedup benchmark. In a separate 4-step stage comparison, moving VAE decoding from CPU to Vulkan reduced the measured VAE stage from about 24.9 seconds to about 2.7 seconds. These are local measurements, not performance guarantees; results depend heavily on checkpoint, resolution, steps, driver, and VRAM pressure.

The native kernels accumulate SIMD partial sums in vector lanes before reducing them, and the Qwen3.5 Numba path fuses recurrent-state operations to avoid an outer-product temporary and reduce full-state passes. In separate short A/B tests, the fused recurrent step improved measured decode throughput by about 6.5% on the main Qwen3.5 checkpoint and 8.7% on the vision-tagged checkpoint. The longer-term Qwen3.5 target is 6–7 tok/s and has not yet been reached.

## Known limitations

- Qwen3.5 is currently text-only, including the checkpoint whose name contains `vision`.
- Image generation currently targets SDXL through `stable-diffusion.cpp`; other diffusion architectures are not claimed to be supported.
- Vulkan hybrid mode assigns diffusion and VAE to the GPU and text encoders to the CPU; it does not provide CPU/GPU layer-level parallelism within one diffusion graph.
- Video, multimodal image input, general MoE execution, IQ quantization, and a general Jinja chat-template engine are not implemented.
- Some quantization types use a bounded dequantization cache and portable fallback paths.
- Prompt prefill can be a significant part of total response time, especially for long conversation histories.
- Single-token generation remains slower than mature inference runtimes. Model outputs still need broader numerical, multilingual, and long-context validation; ZAYA in particular can produce awkward or incorrect Russian prose despite reasonable decode speed.
- This project is under active development; performance and model coverage may change.

## Design principle

Architecture support is explicit. `engine/architectures.py` maps GGUF architecture metadata to a dedicated implementation; unknown architectures fail clearly instead of being silently treated as Llama. New model families should receive their own implementation when their block structure differs.

## License

Check the repository's license file before redistributing this project or its components. Model weights are separate and remain subject to their own licenses.
