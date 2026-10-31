"""CPU-first UI for the separate in-project generative-model engine."""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QPlainTextEdit, QSpinBox, QVBoxLayout, QWidget,
)
from engine.generative import GenerativeEngine, GenerativeModelError


class ImageGenerationWorker(QObject):
    status = Signal(str)
    completed = Signal(str, float)
    failed = Signal(str)
    done = Signal()

    def __init__(self, engine, model_path, prompt, output_path, options):
        super().__init__()
        self.engine = engine
        self.model_path = Path(model_path)
        self.prompt = prompt
        self.output_path = Path(output_path)
        self.options = options
        self.stop_event = threading.Event()

    def cancel(self):
        self.stop_event.set()

    @Slot()
    def run(self):
        started = time.perf_counter()
        try:
            self.status.emit(
                "Проверка SDXL-модели и runtime (CPU + AMD Vulkan)…"
                if self.options.get("use_vulkan") else "Проверка SDXL-модели и runtime (CPU)…"
            )
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.engine.generate(
                self.model_path, self.prompt, self.output_path,
                progress=self.status.emit, cancelled=self.stop_event.is_set,
                **self.options,
            )
            if self.stop_event.is_set():
                raise RuntimeError("Генерация отменена пользователем.")
            if not self.output_path.is_file() or self.output_path.stat().st_size == 0:
                raise RuntimeError("Движок не создал файл изображения.")
            self.completed.emit(str(self.output_path), time.perf_counter() - started)
        except Exception as exc:
            logging.getLogger(__name__).exception("Image generation failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.done.emit()


class ImageGenerationPage(QWidget):
    """Image generation tab backed by the independent generative engine."""
    status_changed = Signal(str)

    def __init__(self, settings, project_root, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.project_root = Path(project_root).resolve()
        self.engine = GenerativeEngine(self.project_root)
        self.thread = None
        self.worker = None
        self.output_path = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        model_row = QHBoxLayout()
        detected = self.engine.find_sdxl_model()
        remembered = settings.value("image_model_path", "", type=str)
        initial_model = remembered if remembered and Path(remembered).is_file() else str(detected or "")
        self.model_path = QLineEdit(initial_model)
        self.model_path.setPlaceholderText("Путь к модели генерации изображений (.gguf, .safetensors, .ckpt)")
        model_button = QPushButton("Выбрать модель…")
        model_button.clicked.connect(self.choose_model)
        model_row.addWidget(self.model_path, 1)
        model_row.addWidget(model_button)
        form.addRow("Модель:", model_row)

        self.prompt = QPlainTextEdit()
        self.prompt.setMaximumHeight(90)
        self.prompt.setPlaceholderText("Опишите изображение. Например: a red fox in a snowy forest, cinematic lighting")
        form.addRow("Промпт:", self.prompt)
        self.negative_prompt = QLineEdit()
        self.negative_prompt.setPlaceholderText("Необязательный negative prompt")
        form.addRow("Исключить:", self.negative_prompt)

        options = QHBoxLayout()
        self.width = self._spin(64, 2048, 512, 64)
        self.height = self._spin(64, 2048, 512, 64)
        self.steps = self._spin(1, 150, 20, 1)
        self.seed = self._spin(-1, 2147483647, 42, 1)
        for label, control in (("Ширина", self.width), ("Высота", self.height),
                               ("Шаги", self.steps), ("Seed (-1 — случайный)", self.seed)):
            options.addWidget(QLabel(label))
            options.addWidget(control)
        form.addRow("Параметры:", options)
        layout.addLayout(form)

        action_row = QHBoxLayout()
        self.generate_button = QPushButton("Сгенерировать изображение (CPU)")
        self.generate_button.clicked.connect(self.generate)
        self.cancel_button = QPushButton("Остановить")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_generation)
        self.open_output_button = QPushButton("Открыть результат…")
        self.open_output_button.clicked.connect(self.open_output)
        action_row.addWidget(self.generate_button)
        action_row.addWidget(self.cancel_button)
        action_row.addWidget(self.open_output_button)
        layout.addLayout(action_row)

        self.status_label = QLabel(
            f"Отдельный движок генерации · CPU. Модель: {initial_model or 'не найдена в ' + str(self.engine.models_dir)}. "
            "Выбирать внешний исполняемый файл не требуется."
        )
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.preview = QLabel("Здесь появится готовое изображение")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(260)
        self.preview.setStyleSheet("QLabel { background: rgba(12, 15, 21, 175); color: white; border: 1px solid rgba(255,255,255,55); }")
        layout.addWidget(self.preview, 1)

    @staticmethod
    def _spin(minimum, maximum, value, step):
        control = QSpinBox()
        control.setRange(minimum, maximum)
        control.setValue(value)
        control.setSingleStep(step)
        return control

    def choose_model(self):
        current_text = self.model_path.text().strip()
        current = Path(current_text).expanduser() if current_text else self.engine.models_dir
        start_dir = str(current if current.is_dir() else current.parent)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите модель генерации изображений",
            start_dir,
            "Модели изображений (*.gguf *.safetensors *.ckpt);;Все файлы (*.*)",
        )
        if path:
            self.model_path.setText(path)
            self.settings.setValue("image_model_path", path)

    def generate(self):
        if self.thread is not None:
            return
        model = Path(self.model_path.text().strip().strip('"')).expanduser()
        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Нет промпта", "Сначала опишите изображение.")
            return
        try:
            self.engine.inspect_model(model)
        except GenerativeModelError as exc:
            QMessageBox.warning(self, "Модель не подходит", str(exc))
            return
        self.settings.setValue("image_model_path", str(model.resolve()))
        output_dir = self.project_root / "generated_images"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"image_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000:06d}.png"
        options = {
            "negative_prompt": self.negative_prompt.text(),
            "width": self.width.value(), "height": self.height.value(),
            "steps": self.steps.value(), "seed": self.seed.value(),
            # Let stable-diffusion.cpp select physical cores (threads=0).
            "threads": 0,
            "use_vulkan": self.settings.value("image_vulkan_enabled", False, type=bool),
        }
        self.generate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status_label.setText("Движок проверяет модель и компоненты SDXL…")
        self.status_changed.emit(self.status_label.text())
        self.thread = QThread(self)
        self.worker = ImageGenerationWorker(self.engine, model.resolve(), prompt, output, options)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.status.connect(self.status_label.setText, Qt.ConnectionType.QueuedConnection)
        self.worker.completed.connect(self.generation_completed, Qt.ConnectionType.QueuedConnection)
        self.worker.failed.connect(self.generation_failed, Qt.ConnectionType.QueuedConnection)
        self.worker.done.connect(self.thread.quit)
        self.worker.done.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.generation_finished, Qt.ConnectionType.QueuedConnection)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @Slot(str, float)
    def generation_completed(self, path, elapsed):
        self.output_path = Path(path)
        self.show_image()
        self.status_label.setText(f"Готово за {elapsed:.1f} с · {path}")
        self.status_changed.emit(self.status_label.text())

    @Slot(str)
    def generation_failed(self, message):
        cancelled = "отменена пользователем" in message.casefold()
        self.status_label.setText("Генерация остановлена." if cancelled else "Ошибка генерации. Подробности — в Консоли / Логи.")
        self.status_changed.emit(self.status_label.text())
        if not cancelled:
            QMessageBox.critical(self, "Ошибка генерации изображения", message)

    def cancel_generation(self):
        if self.worker is not None:
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Остановка генерации…")
            self.worker.cancel()

    @Slot()
    def generation_finished(self):
        self.thread = None
        self.worker = None
        self.generate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def shutdown(self):
        """Request cancellation before the owning window is destroyed."""
        if self.worker is not None and self.thread is not None and self.thread.isRunning():
            self.worker.cancel()
            self.thread.quit()
            self.thread.wait(7000)

    def show_image(self):
        if not self.output_path:
            return
        pixmap = QPixmap(str(self.output_path))
        if pixmap.isNull():
            self.preview.setText(f"Изображение сохранено: {self.output_path}")
            return
        self.preview.setPixmap(pixmap.scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.output_path:
            self.show_image()

    def open_output(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть изображение", str(self.project_root / "generated_images"), "Изображения (*.png *.jpg *.jpeg *.webp *.bmp)")
        if path:
            self.output_path = Path(path)
            self.show_image()
