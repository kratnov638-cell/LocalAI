"""Model discovery and runtime boundary for local image generation.

This package deliberately does not route diffusion models through the text GGUF
inference stack. The first supported model family is SDXL. A GGUF U-Net file
is only one component of an SDXL pipeline; text encoders and a VAE are also
required unless the chosen runtime/model bundle provides them internally.
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from engine.gguf_parser import GGUFFile


class GenerativeModelError(RuntimeError):
    """Actionable model or runtime error shown by the image-generation UI."""


@dataclass(frozen=True)
class ModelInfo:
    path: Path
    architecture: str
    tensor_count: int
    size_bytes: int


class GenerativeEngine:
    """Owns generative-model discovery, validation and inference dispatch.

    Runtime lookup is automatic: the user never has to choose an executable.
    The runtime boundary is isolated here so diffusion inference can evolve
    independently from ``engine.model`` (the language-model implementation).
    """

    SUPPORTED_ARCHITECTURES = {"sdxl"}
    RUNTIME_RELATIVE_PATHS = (
        Path("engine/generative/bin/sd-cli.exe"),
        Path("engine/generative/bin/sd-cli"),
        Path("engine/generative/bin/sd-master-77b72a2-bin-win-cpu-x64/sd-cli.exe"),
        Path("engine/generative/bin/sd-master-77b72a2-bin-win-cpu-x64/sd-cli"),
    )

    def __init__(self, project_root: str | Path, models_dir: str | Path | None = None):
        self.project_root = Path(project_root).resolve()
        self.models_dir = Path(models_dir).resolve() if models_dir else self._default_models_dir()

    def _default_models_dir(self) -> Path:
        configured = os.environ.get("LOCALAI_MODELS_DIR")
        if configured:
            return Path(configured).expanduser().resolve()
        windows_default = Path(r"C:\LLM")
        if windows_default.is_dir():
            return windows_default
        return self.project_root / "models"

    def discover_models(self) -> list[Path]:
        """Find local diffusion checkpoints without scanning unrelated directories."""
        if not self.models_dir.is_dir():
            return []
        return sorted(
            (p for p in self.models_dir.iterdir()
             if p.is_file() and p.suffix.casefold() in {".gguf", ".safetensors", ".ckpt"}),
            key=lambda p: p.name.casefold(),
        )

    def find_sdxl_model(self) -> Path | None:
        """Find likely SDXL files without opening every unrelated LLM checkpoint."""
        candidates = [p for p in self.discover_models() if "sdxl" in p.name.casefold() or "sd_xl" in p.name.casefold()]
        for path in candidates:
            if path.suffix.casefold() in {".safetensors", ".ckpt"}:
                return path.resolve()
            try:
                with GGUFFile(str(path)) as gguf:
                    if str(gguf.metadata.get("general.architecture", "")).casefold() == "sdxl":
                        return path.resolve()
            except Exception:
                continue
        return candidates[0].resolve() if candidates else None

    def inspect_model(self, model_path: str | Path) -> ModelInfo:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise GenerativeModelError(f"Файл модели не найден: {path}")
        if path.suffix.casefold() in {".safetensors", ".ckpt"}:
            if path.stat().st_size < 1024 * 1024:
                raise GenerativeModelError("Файл checkpoint слишком мал и не похож на полную модель диффузии.")
            return ModelInfo(path, "sdxl-checkpoint", 0, path.stat().st_size)
        if path.suffix.casefold() != ".gguf":
            raise GenerativeModelError("Поддерживаются diffusion checkpoint .safetensors/.ckpt и standalone-модели .gguf.")
        try:
            with GGUFFile(str(path)) as gguf:
                architecture = str(gguf.metadata.get("general.architecture", "")).casefold()
                tensor_count = len(gguf.tensors)
        except Exception as exc:
            raise GenerativeModelError(f"Не удалось прочитать GGUF-модель: {exc}") from exc
        if architecture not in self.SUPPORTED_ARCHITECTURES:
            raise GenerativeModelError(
                f"Архитектура {architecture or 'неизвестна'} не поддерживается движком генерации. "
                "Сейчас подключается SDXL; языковые GGUF-модели сюда не подходят."
            )
        return ModelInfo(path, architecture, tensor_count, path.stat().st_size)

    def runtime_path(self) -> Path | None:
        """Find a project-bundled runtime; never ask the user to select it."""
        for relative in self.RUNTIME_RELATIVE_PATHS:
            candidate = self.project_root / relative
            if candidate.is_file():
                return candidate.resolve()
        # Allow a packaged deployment to put the runtime on PATH, while keeping
        # the executable invisible from the UI and avoiding arbitrary path input.
        found = shutil.which("sd-cli.exe") or shutil.which("sd-cli")
        return Path(found).resolve() if found else None

    def vulkan_device(self) -> str | None:
        """Return the first Vulkan backend device exposed by the bundled runtime."""
        runtime = self.runtime_path()
        if runtime is None:
            return None
        try:
            result = subprocess.run(
                [str(runtime), "--list-devices"], cwd=str(runtime.parent),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            name, separator, description = line.partition("\t")
            if separator and name.strip().casefold().startswith("vulkan"):
                return f"{name.strip()} — {description.strip()}"
        return None

    @staticmethod
    def _backend_args(use_vulkan: bool) -> list[str]:
        """Run the expensive diffusion and VAE stages on Vulkan, keep text encoding on CPU."""
        if use_vulkan:
            return [
                "--backend", "diffusion=vulkan0,te=cpu,vae=vulkan0",
                "--max-vram", "vulkan0=3.2", "--auto-fit", "on",
            ]
        return ["--backend", "cpu"]

    def generate(self, model_path: str | Path, prompt: str, output_path: str | Path,
                 *, negative_prompt: str = "", width: int = 512, height: int = 512,
                 steps: int = 20, seed: int = 42, threads: int | None = None,
                 use_vulkan: bool = False,
                 progress: Callable[[str], None] | None = None,
                 cancelled: Callable[[], bool] | None = None) -> Path:
        """Validate and dispatch generation through the configured runtime.

        The UI does not expose runtime selection. A missing runtime or missing
        SDXL pipeline components fails explicitly instead of pretending that a
        U-Net GGUF alone is a complete text-to-image model.
        """
        info = self.inspect_model(model_path)
        if not prompt.strip():
            raise GenerativeModelError("Введите описание изображения (промпт).")
        runtime = self.runtime_path()
        if runtime is None:
            expected = self.project_root / "engine/generative/bin"
            raise GenerativeModelError(
                "Встроенный runtime генерации ещё не установлен. "
                f"Движок ищет его автоматически в {expected}; выбирать exe в интерфейсе не нужно. "
                "Поддержка GGUF-парсера сама по себе не запускает диффузионную модель."
            )
        is_full_checkpoint = info.path.suffix.casefold() in {".safetensors", ".ckpt"}
        if use_vulkan and not self.vulkan_device():
            raise GenerativeModelError(
                "Включён Vulkan, но stable-diffusion.cpp не обнаруживает Vulkan backend. "
                "Проверьте сборку ggml-vulkan.dll и драйвер AMD; переключатель можно отключить в Настройках."
            )
        components = (None, None, None) if is_full_checkpoint else self._find_sdxl_components(info.path.parent)
        if not is_full_checkpoint and (not components or not all(components)):
            clip_l = components[0] if components else None
            clip_g = components[1] if components else None
            vae = components[2] if components else None
            missing = []
            if not clip_l:
                missing.append("CLIP-L (около 247 МБ)")
            if not clip_g:
                missing.append("CLIP-G (около 1,39 ГБ)")
            if not vae:
                missing.append("VAE (около 167 МБ)")
            raise GenerativeModelError(
                "Не найдены обязательные компоненты SDXL: " + ", ".join(missing) + ".\n\n"
                "Текущая GGUF содержит только UNet; это не ошибка запуска CLI. Скачайте "
                "совместимые файлы из репозитория модели https://huggingface.co/HyperX-Sentience/SDXL-GGUF "
                "и сохраните их в C:\\LLM\\sdxl-components\\clip\\ и C:\\LLM\\sdxl-components\\vae\\:\n"
                "  CLIP-L: https://huggingface.co/HyperX-Sentience/SDXL-GGUF/resolve/main/clip/sdxl_clip_l.safetensors\n"
                "  CLIP-G: https://huggingface.co/HyperX-Sentience/SDXL-GGUF/resolve/main/clip/sdxl_clip_g.safetensors\n"
                "  VAE:    https://huggingface.co/HyperX-Sentience/SDXL-GGUF/resolve/main/vae/sdxl_vae.safetensors\n\n"
                "Либо задайте пути переменными LOCALAI_SDXL_CLIP_L, LOCALAI_SDXL_CLIP_G и "
                "LOCALAI_SDXL_VAE. Файлы намеренно не скачиваются автоматически."
            )
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        # Read prompt text from UTF-8 files to avoid Windows ANSI argv
        # conversion corrupting non-ASCII (for example, Russian) prompts.
        prompt_files: list[Path] = []
        for label, value in (("prompt", prompt.strip()), ("negative-prompt", negative_prompt)):
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", suffix=f".{label}.txt",
                prefix="localai-", dir=str(output.parent), delete=False,
            ) as handle:
                handle.write(value)
                prompt_files.append(Path(handle.name))
        command = [str(runtime), "--mode", "img_gen"]
        if is_full_checkpoint:
            command.extend(["--model", str(info.path)])
        else:
            clip_l, clip_g, vae = components
            command.extend([
                "--diffusion-model", str(info.path),
                "--clip_l", str(clip_l), "--clip_g", str(clip_g), "--vae", str(vae),
            ])
        command.extend(self._backend_args(use_vulkan) + [
            # stable-diffusion.cpp interprets <= 0 as physical CPU cores.
            # Do not force all logical threads: that can oversubscribe this CPU.
            "--threads", str(int(threads) if threads is not None else 0),
            "--rng", "cpu",
            "--prompt-file", str(prompt_files[0]),
            "--negative-prompt-file", str(prompt_files[1]),
            "--width", str(int(width)), "--height", str(int(height)),
            "--steps", str(int(steps)), "--seed", str(int(seed)), "--output", str(output),
        ])
        if width < 64 or height < 64 or width % 8 or height % 8:
            raise GenerativeModelError("Ширина и высота должны быть не меньше 64 и кратны 8.")
        if not 1 <= int(steps) <= 150:
            raise GenerativeModelError("Число шагов должно быть от 1 до 150.")
        if cancelled and cancelled():
            raise GenerativeModelError("Генерация отменена пользователем.")
        if progress:
            progress(
                "Запуск stable-diffusion.cpp: diffusion и VAE — AMD Vulkan, текстовые энкодеры — CPU…"
                if use_vulkan else "Запуск stable-diffusion.cpp (CPU); загрузка компонентов SDXL…"
            )
        logger = __import__("logging").getLogger(__name__)
        logger.info("SDXL command: %s", subprocess.list2cmdline(command))
        try:
            process = subprocess.Popen(
                command, cwd=str(runtime.parent), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                bufsize=1, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise GenerativeModelError(f"Не удалось запустить stable-diffusion.cpp: {exc}") from exc
        lines: queue.Queue[str | None] = queue.Queue()

        def read_output():
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    lines.put(line.rstrip())
            finally:
                lines.put(None)

        reader = threading.Thread(target=read_output, name="sd-cli-output", daemon=True)
        reader.start()
        captured: list[str] = []
        last_status = 0.0
        stream_closed = False
        while process.poll() is None or not stream_closed:
            if cancelled and cancelled() and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise GenerativeModelError("Генерация отменена пользователем.")
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                line = ""
            if line is None:
                stream_closed = True
                continue
            if line:
                captured.append(line)
                logger.info("sd-cli: %s", line)
                now = time.monotonic()
                if progress and now - last_status >= 1.0:
                    progress(line[:240])
                    last_status = now
        reader.join(timeout=1)
        for prompt_file in prompt_files:
            prompt_file.unlink(missing_ok=True)
        log_tail = "\n".join(captured[-35:])
        if process.returncode != 0:
            raise GenerativeModelError(
                f"stable-diffusion.cpp завершился с кодом {process.returncode}.\n{log_tail}"
            )
        if not output.is_file() or output.stat().st_size < 100:
            raise GenerativeModelError(
                f"Runtime завершился без корректного файла изображения.\n{log_tail}"
            )
        try:
            with output.open("rb") as image_file:
                signature = image_file.read(8)
            if signature != b"\x89PNG\r\n\x1a\n":
                raise ValueError("не PNG signature")
            try:
                from PIL import Image
                with Image.open(output) as image:
                    image.verify()
                with Image.open(output) as image:
                    if image.format != "PNG" or image.width != int(width) or image.height != int(height):
                        raise ValueError(f"неожиданный формат/размер: {image.format} {image.size}")
            except ImportError:
                pass
        except Exception as exc:
            raise GenerativeModelError(f"Файл результата не прошёл проверку изображения: {exc}\n{log_tail}") from exc
        if progress:
            progress(f"Изображение проверено: {output.name} ({output.stat().st_size:,} байт)")
        return output

    def _find_sdxl_components(self, folder: Path) -> tuple[Path | None, Path | None, Path | None]:
        """Resolve SDXL text encoders and VAE from explicit env vars or local files."""
        search_dirs = [folder, self.models_dir, self.project_root / "models", self.project_root / "engine/generative/models"]
        search_dirs = list(dict.fromkeys(path.resolve() for path in search_dirs if path.is_dir()))

        def resolve(env_name: str, markers: tuple[str, ...]) -> Path | None:
            configured = os.environ.get(env_name)
            if configured:
                path = Path(configured).expanduser().resolve()
                return path if path.is_file() else None
            for directory in search_dirs:
                for path in sorted(directory.rglob("*"), key=lambda item: str(item).casefold()):
                    if not path.is_file() or path.suffix.casefold() not in {".gguf", ".safetensors", ".ckpt", ".pt", ".bin"}:
                        continue
                    relative = str(path.relative_to(directory)).casefold().replace("\\", "/")
                    if any(marker in relative for marker in markers):
                        if "text_encoder/" in relative and "text_encoder_2/" in relative:
                            continue
                        return path.resolve()
            return None

        clip_l = resolve("LOCALAI_SDXL_CLIP_L", ("clip_l", "clip-l", "text_encoder/", "text_encoder_1"))
        clip_g = resolve("LOCALAI_SDXL_CLIP_G", ("clip_g", "clip-g", "text_encoder_2"))
        vae = resolve("LOCALAI_SDXL_VAE", ("vae",))
        return clip_l, clip_g, vae
