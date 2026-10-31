import struct
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from engine.generative import GenerativeEngine, GenerativeModelError


def _gguf_string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _write_minimal_gguf(path, architecture="sdxl"):
    # GGUF v3, one metadata entry, zero tensors: sufficient for architecture
    # inspection tests without allocating a real multi-gigabyte checkpoint.
    data = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, 1))
    data += _gguf_string("general.architecture") + struct.pack("<I", 8)
    data += _gguf_string(architecture)
    path.write_bytes(data)


class GenerativeEngineTests(unittest.TestCase):
    def test_cpu_backend_remains_default(self):
        self.assertEqual(GenerativeEngine._backend_args(False), ["--backend", "cpu"])

    def test_vulkan_backend_splits_sdxl_modules_between_gpu_and_cpu(self):
        self.assertEqual(
            GenerativeEngine._backend_args(True),
            [
                "--backend", "diffusion=vulkan0,te=cpu,vae=vulkan0",
                "--max-vram", "vulkan0=3.2", "--auto-fit", "on",
            ],
        )

    def test_vulkan_device_is_detected_from_runtime_device_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "engine" / "generative" / "bin" / "sd-cli.exe"
            runtime.parent.mkdir(parents=True)
            runtime.write_bytes(b"test executable placeholder")
            engine = GenerativeEngine(root, root)
            completed = __import__("subprocess").CompletedProcess(
                args=[], returncode=0,
                stdout="CPU\tIntel CPU\nVulkan0\tAMD Radeon RX 6400\n",
            )
            with patch("engine.generative.engine.subprocess.run", return_value=completed):
                self.assertEqual(engine.vulkan_device(), "Vulkan0 — AMD Radeon RX 6400")

    def test_finds_bundled_sd_cli_in_bin_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "engine" / "generative" / "bin" / "sd-cli.exe"
            runtime.parent.mkdir(parents=True)
            runtime.write_bytes(b"test executable placeholder")
            self.assertEqual(GenerativeEngine(root, root).runtime_path(), runtime.resolve())

    def test_discovers_sdxl_model_by_filename_without_scanning_other_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "LLM"
            models.mkdir()
            model = models / "sdxl_base.gguf"
            _write_minimal_gguf(model)
            (models / "qwen.gguf").write_bytes(b"not read by discovery")
            engine = GenerativeEngine(root, models)
            self.assertEqual(engine.find_sdxl_model(), model.resolve())

    def test_inspects_sdxl_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "sdxl.gguf"
            _write_minimal_gguf(model)
            info = GenerativeEngine(root, root).inspect_model(model)
            self.assertEqual(info.architecture, "sdxl")
            self.assertEqual(info.tensor_count, 0)

    def test_accepts_full_safetensors_checkpoint_without_companion_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "sd_xl_base_1.0.safetensors"
            model.write_bytes(b"0" * (1024 * 1024))
            info = GenerativeEngine(root, root).inspect_model(model)
            self.assertEqual(info.architecture, "sdxl-checkpoint")
            self.assertEqual(info.size_bytes, model.stat().st_size)

    def test_discovers_full_safetensors_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "sd_xl_base_1.0.safetensors"
            model.write_bytes(b"0" * (1024 * 1024))
            self.assertEqual(GenerativeEngine(root, root).find_sdxl_model(), model.resolve())

    def test_rejects_language_model_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "llama.gguf"
            _write_minimal_gguf(model, "llama")
            with self.assertRaisesRegex(GenerativeModelError, "не поддерживается"):
                GenerativeEngine(root, root).inspect_model(model)

    def test_missing_model_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = GenerativeEngine(tmp, tmp)
            with self.assertRaisesRegex(GenerativeModelError, "не найден"):
                engine.inspect_model(Path(tmp) / "missing.gguf")

    def test_model_is_detected_in_real_llm_directory_when_present(self):
        model = Path(r"C:\LLM\sdxl_base_1.0_Q4_K_S.gguf")
        if not model.is_file():
            self.skipTest("Local SDXL checkpoint is not available on this machine")
        engine = GenerativeEngine(Path(__file__).resolve().parents[1])
        info = engine.inspect_model(model)
        self.assertEqual(info.architecture, "sdxl")
        self.assertGreater(info.tensor_count, 100)

    def test_generation_explains_missing_sdxl_components_without_running_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "sdxl_base.gguf"
            _write_minimal_gguf(model)
            runtime = root / "engine" / "generative" / "bin" / "sd-cli.exe"
            runtime.parent.mkdir(parents=True)
            runtime.write_bytes(b"test executable placeholder")
            engine = GenerativeEngine(root, root)
            with patch.dict("os.environ", {
                "LOCALAI_SDXL_CLIP_L": "", "LOCALAI_SDXL_CLIP_G": "", "LOCALAI_SDXL_VAE": "",
            }, clear=False):
                with self.assertRaisesRegex(GenerativeModelError, "CLIP-L.*CLIP-G.*VAE"):
                    engine.generate(model, "a test image", root / "out.png")


if __name__ == "__main__":
    unittest.main()
