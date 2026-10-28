"""PySide6 chat UI for the in-repository NumPy inference engine."""
import logging
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
from PySide6.QtCore import QSettings
from PySide6.QtGui import QAction, QActionGroup, QPixmap
from PySide6.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QSpinBox, QTextEdit,
    QPlainTextEdit, QTabWidget, QVBoxLayout, QWidget, QMenu, QToolButton,
)

from engine.gguf_parser import GGUFFile
from engine.session import InferenceSession


class _QtLogHandler(logging.Handler):
    """Deliver Python log records to the GUI using a queued Qt signal."""
    def __init__(self, signal):
        super().__init__()
        self.signal = signal
        self.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s', '%H:%M:%S'))

    def emit(self, record):
        try:
            self.signal.emit(self.format(record))
        except RuntimeError:
            # The window may already be closing while a worker reports shutdown.
            pass


def format_chat(messages, metadata):
    """Format the small supported chat subset for common instruct GGUFs."""
    template = str(metadata.get('tokenizer.chat_template', ''))
    is_qwen = '<|im_start|>' in template or str(metadata.get('tokenizer.ggml.pre', '')).startswith(('qwen2', 'qwen3', 'qwen35'))
    is_llama3 = '<|start_header_id|>' in template or 'llama-bpe' in str(metadata.get('tokenizer.ggml.pre', ''))
    if is_qwen:
        qwen35 = str(metadata.get('tokenizer.ggml.pre', '')).startswith('qwen35') or '<think>' in template
        prompt_messages = list(messages)
        if qwen35 and not any(m.get('role') == 'system' for m in prompt_messages):
            language = str(metadata.get('_localai_language', 'ru')).lower()
            prompt_messages.insert(0, {
                'role': 'system',
                'content': (
                    'Answer exclusively in English unless the user explicitly asks for another language. '
                    'Use natural, grammatically correct English and do not mix languages.'
                    if language == 'en' else
                    'Отвечай исключительно на русском языке, если пользователь явно не попросил '
                    'ответить на другом языке. Не используй украинские слова и формулировки '
                    'и не смешивай русский с другими языками. Пиши естественно и грамотно по-русски.'
                ),
            })
        prompt = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in prompt_messages)
        # Match Qwen3.5's embedded chat template with enable_thinking=False.
        # Closing the think block in the prompt asks for a user-facing answer
        # directly instead of streaming hidden reasoning as chat text.
        if qwen35:
            return prompt + '<|im_start|>assistant\n<think>\n\n</think>\n\n'
        return prompt + '<|im_start|>assistant\n'
    if is_llama3:
        prompt = ''.join(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content'].strip()}<|eot_id|>" for m in messages)
        # Llama 3's embedded template starts with bos_token. GGUF exporters may
        # set add_bos_token=false because the chat template inserts it itself;
        # omitting it here makes otherwise valid prompts produce poor/garbled
        # generations on both the Q4_K and Q8_0 Saiga checkpoints.
        tokens = metadata.get('tokenizer.ggml.tokens', ())
        bos_id = int(metadata.get('tokenizer.ggml.bos_token_id', 128000))
        bos_token = tokens[bos_id] if 0 <= bos_id < len(tokens) else '<|begin_of_text|>'
        return bos_token + prompt + '<|start_header_id|>assistant<|end_header_id|>\n\n'
    prompt = ''.join(f"{m['role'].capitalize()}: {m['content']}\n" for m in messages)
    return prompt + 'Assistant: '


class GenerationWorker(QObject):
    text = Signal(str)
    status = Signal(str)
    error = Signal(str)
    session_ready = Signal(object)
    done = Signal()

    def __init__(self, path, prompt, settings, stop_event, session=None):
        super().__init__()
        self.path, self.prompt, self.settings, self.stop_event = path, prompt, settings, stop_event
        self.session = session

    @Slot()
    def run(self):
        emitted = 0
        try:
            if self.session is None:
                logging.getLogger(__name__).info('Opening model: %s', self.path)
                self.status.emit('Loading model and tokenizer…' if self.settings.get('language') == 'en' else 'Загрузка модели и токенизатора…')
                self.session = InferenceSession(self.path)
                self.session_ready.emit(self.session)
            tokenizer = self.session.tokenizer
            model = self.session.model
            english = self.settings.get('language', 'ru') == 'en'
            model_name = str(self.session.gguf.metadata.get('general.name', '')).lower()
            ids = tokenizer.encode(self.prompt)
            self.status.emit(f'Prompt: {len(ids)} tokens' if english else f'Промпт: {len(ids)} токенов')
            generation_start = time.perf_counter()
            logging.getLogger(__name__).info('Generation started; prompt_tokens=%d', len(ids))

            def emit(piece):
                nonlocal emitted
                # Token decoding can emit an empty fragment for partial UTF-8,
                # and a final decoder flush can emit text without a new token.
                # Count sampled tokens from the model, not UI text chunks.
                emitted = int(getattr(model, 'last_generated_tokens', emitted + 1))
                self.text.emit(piece)
                elapsed = max(time.perf_counter() - generation_start, 1e-6)
                prefill_s = max(0.0, float(getattr(model, 'last_prefill_seconds', 0.0)))
                decode_elapsed = max(elapsed - prefill_s, 1e-6)
                self.status.emit(
                    f'Generating | {emitted / decode_elapsed:.2f} tok/s decode | prefill {prefill_s:.1f}s'
                    if english else
                    f'\u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f | {emitted / decode_elapsed:.2f} \u0442\u043e\u043a/\u0441 | \u043f\u0440\u0435\u0444\u0438\u043b\u043b {prefill_s:.1f} \u0441'
                )

            if hasattr(model, 'progress_callback'):
                model.progress_callback = self.status.emit
            try:
                generation_settings = {key: value for key, value in self.settings.items() if key != 'language'}
                if 'saiga_llama3_8b' in model_name:
                    # Match the known-good Saiga/KoboldCpp sampling profile.
                    # Very low temperature plus aggressive repetition penalty
                    # made the old incorrect RoPE path look even less fluent.
                    generation_settings.setdefault('top_k', 100)
                    generation_settings.setdefault('repetition_penalty', 1.05)
                model.generate(ids, tokenizer, self.stop_event, emit, **generation_settings)
                profiler = getattr(model, 'decode_profiler', None)
                profile = profiler.report() if profiler is not None else None
                if profile is not None:
                    top_matrices = ','.join(
                        f"{item['name']}:{item['seconds']:.3f}s/{item['calls']}"
                        for item in profile.get('top_matrices', ())
                    ) or 'none'
                    logging.getLogger(__name__).info(
                        'Decode profile (instrumented; excludes prefill): forward_s=%.3f forward_calls=%d sampling_s=%.3f non_mat_s=%.3f mat_s=%s mat_calls=%s top_matrices=%s',
                        profile['forward_s'], profile['forward_calls'], profile['sampling_s'],
                        profile['non_mat_s'],
                        ','.join(f'{key}:{value:.3f}' for key, value in profile['mat_s'].items()),
                        ','.join(f'{key}:{value}' for key, value in profile['mat_calls'].items()),
                        top_matrices,
                    )
            finally:
                if hasattr(model, 'progress_callback'):
                    model.progress_callback = None
            elapsed = max(time.perf_counter() - generation_start, 1e-6)
            prefill_s = max(0.0, float(getattr(model, 'last_prefill_seconds', 0.0)))
            decode_elapsed = max(elapsed - prefill_s, 1e-6)
            decode_rate = emitted / decode_elapsed
            total_rate = emitted / elapsed
            self.status.emit(
                f'Finished | {emitted} tokens | {decode_rate:.2f} tok/s decode | prefill {prefill_s:.1f}s | total {elapsed:.1f}s'
                if english else
                f'\u0413\u043e\u0442\u043e\u0432\u043e | {emitted} \u0442\u043e\u043a\u0435\u043d\u043e\u0432 | {decode_rate:.2f} \u0442\u043e\u043a/\u0441 | \u043f\u0440\u0435\u0444\u0438\u043b\u043b {prefill_s:.1f} \u0441 | \u0432\u0441\u0435\u0433\u043e {elapsed:.1f} \u0441'
            )
            logging.getLogger(__name__).info(
                'Generation finished; output_tokens=%d; prefill_s=%.3f; decode_s=%.3f; decode_tok_s=%.3f; total_tok_s=%.3f',
                emitted, prefill_s, decode_elapsed, decode_rate, total_rate,
            )
        except Exception as exc:
            logging.getLogger(__name__).exception('Inference worker failed')
            self.error.emit(f'{type(exc).__name__}: {exc}')
        finally:
            self.done.emit()


class MainWindow(QMainWindow):
    log_message = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('LocalAI — custom NumPy GGUF engine')
        self.resize(1000, 760)
        self.model_path = None
        self.metadata = None
        self.session = None
        self.messages = []
        self.stop_event = threading.Event()
        self.worker = None
        self.thread = None
        self.current_answer = ''
        self.settings = QSettings('LocalAI', 'LocalAI')
        self.language = self.settings.value('language', 'ru', type=str)
        if self.language not in ('ru', 'en'):
            self.language = 'ru'
        self.default_background = Path(__file__).resolve().parent / 'assets' / 'backgrounds' / 'default.jpg'
        saved_background = self.settings.value('background_path', '', type=str)
        self.background_path = saved_background if saved_background and Path(saved_background).is_file() else str(self.default_background)
        self._log_handler = _QtLogHandler(self.log_message)
        logging.getLogger().addHandler(self._log_handler)
        logging.getLogger().setLevel(logging.INFO)
        self.log_message.connect(self.append_log, Qt.ConnectionType.QueuedConnection)

        root = QWidget()
        root.setObjectName('mainRoot')
        root.setStyleSheet('#mainRoot { background: transparent; }')
        self.setCentralWidget(root)
        self.background_label = QLabel(root)
        self.background_label.setObjectName('backgroundImage')
        self.background_label.setScaledContents(False)
        self.background_label.lower()
        self._background_pixmap = QPixmap()
        self.set_background(self.background_path, persist=False)
        root_layout = QVBoxLayout(root)
        top_row = QHBoxLayout()
        self.menu_button = QToolButton()
        self.menu_button.setText('☰')
        self.menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu = QMenu(self.menu_button)
        self.menu_button.setMenu(self.menu)
        self.settings_menu = self.menu.addMenu('Настройки')
        self.language_menu = self.settings_menu.addMenu('Язык')
        self.language_group = QActionGroup(self)
        self.language_group.setExclusive(True)
        self.language_actions = {}
        for code, title in (('ru', 'Русский'), ('en', 'English')):
            action = QAction(title, self, checkable=True)
            action.setChecked(self.language == code)
            action.triggered.connect(lambda checked=False, lang=code: self.set_language(lang))
            self.language_group.addAction(action)
            self.language_menu.addAction(action)
            self.language_actions[code] = action
        self.settings_menu.addSeparator()
        self.background_menu = self.settings_menu.addMenu('Фон')
        self.default_background_action = QAction('Стандартный фон', self)
        self.default_background_action.triggered.connect(self.restore_default_background)
        self.background_menu.addAction(self.default_background_action)
        self.choose_background_action = QAction('Выбрать изображение…', self)
        self.choose_background_action.triggered.connect(self.choose_background)
        self.background_menu.addAction(self.choose_background_action)
        top_row.addWidget(self.menu_button, 0, Qt.AlignmentFlag.AlignLeft)
        top_row.addStretch(1)
        root_layout.addLayout(top_row)
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet('QTabWidget::pane { background: rgba(20, 24, 32, 190); border: 1px solid rgba(255,255,255,45); } QTabBar::tab { background: rgba(35, 40, 50, 210); color: white; padding: 7px 12px; } QTabBar::tab:selected { background: rgba(75, 85, 105, 230); }')
        root_layout.addWidget(self.tabs, 1)

        chat_page = QWidget()
        layout = QVBoxLayout(chat_page)
        model_row = QHBoxLayout()
        self.model_label = QLabel('No model selected')
        self.model_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.choose_button = QPushButton('Open GGUF model…')
        self.choose_button.clicked.connect(self.choose_model)
        model_row.addWidget(self.model_label, 1)
        model_row.addWidget(self.choose_button)
        layout.addLayout(model_row)

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setStyleSheet('QTextEdit { background: rgba(12, 15, 21, 175); color: white; border: 1px solid rgba(255,255,255,55); }')
        self.chat.setPlaceholderText('Select a supported GGUF model, then send a message.')
        layout.addWidget(self.chat, 1)

        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setStyleSheet('QLineEdit { background: rgba(12, 15, 21, 205); color: white; border: 1px solid rgba(255,255,255,70); padding: 6px; }')
        self.input.setPlaceholderText('Message the local model…')
        self.input.returnPressed.connect(self.send)
        self.send_button = QPushButton('Send')
        self.send_button.clicked.connect(self.send)
        input_row.addWidget(self.input, 1)
        input_row.addWidget(self.send_button)
        layout.addLayout(input_row)

        settings = QHBoxLayout()
        self._temperature_user_set = False
        self._top_p_user_set = False
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0, 2); self.temperature.setSingleStep(.1); self.temperature.setValue(.8)
        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(.05, 1); self.top_p.setSingleStep(.05); self.top_p.setValue(.95)
        self.temperature.valueChanged.connect(lambda _value: setattr(self, '_temperature_user_set', True))
        self.top_p.valueChanged.connect(lambda _value: setattr(self, '_top_p_user_set', True))
        self.max_new = QSpinBox()
        self.max_new.setRange(1, 4096); self.max_new.setValue(256)
        self.temperature_label = QLabel('Temperature'); settings.addWidget(self.temperature_label); settings.addWidget(self.temperature)
        self.top_p_label = QLabel('Top-p'); settings.addWidget(self.top_p_label); settings.addWidget(self.top_p)
        self.max_new_label = QLabel('Max new tokens'); settings.addWidget(self.max_new_label); settings.addWidget(self.max_new)
        self.stop_button = QPushButton('Stop'); self.stop_button.clicked.connect(self.stop_generation)
        self.new_button = QPushButton('New chat'); self.new_button.clicked.connect(self.new_chat)
        settings.addWidget(self.stop_button); settings.addWidget(self.new_button)
        layout.addLayout(settings)
        self.tabs.addTab(chat_page, 'Chat')

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet('QPlainTextEdit { background: rgba(12, 15, 21, 185); color: white; border: 1px solid rgba(255,255,255,55); }')
        self.log_view.setMaximumBlockCount(3000)
        self.log_view.setPlaceholderText('Diagnostics, errors and generation status appear here.')
        self.tabs.addTab(self.log_view, 'Console / Logs')

        self.status_label = QLabel('Ready · custom NumPy GGUF inference')
        self.statusBar().addWidget(self.status_label)
        self.apply_language()

    def set_background(self, path, persist=True):
        """Set a cover-scaled background, cropping overflow rather than stretching."""
        candidate = Path(path).expanduser()
        pixmap = QPixmap(str(candidate))
        if pixmap.isNull():
            if persist:
                QMessageBox.warning(self, 'Background', 'The selected image could not be opened.')
            return False
        self.background_path = str(candidate.resolve())
        self._background_pixmap = pixmap
        if persist:
            self.settings.setValue('background_path', self.background_path)
        self._update_background_geometry()
        return True

    def _update_background_geometry(self):
        if not hasattr(self, 'background_label'):
            return
        size = self.centralWidget().size()
        self.background_label.setGeometry(0, 0, size.width(), size.height())
        if self._background_pixmap.isNull() or size.width() <= 0 or size.height() <= 0:
            self.background_label.clear()
            return
        scaled = self._background_pixmap.scaled(
            size, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = max(0, (scaled.width() - size.width()) // 2)
        y = max(0, (scaled.height() - size.height()) // 2)
        self.background_label.setPixmap(scaled.copy(x, y, size.width(), size.height()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_background_geometry()

    def choose_background(self):
        start = str(Path(self.background_path).parent) if Path(self.background_path).exists() else str(self.default_background.parent)
        path, _ = QFileDialog.getOpenFileName(
            self, 'Choose background image' if self.language == 'en' else 'Выберите изображение фона',
            start, 'Images (*.jpg *.jpeg *.png *.bmp *.webp)',
        )
        if path:
            self.set_background(path)

    def restore_default_background(self):
        if self.set_background(str(self.default_background)):
            self.settings.remove('background_path')

    def set_language(self, language):
        if language not in ('ru', 'en'):
            return
        self.language = language
        self.settings.setValue('language', language)
        for code, action in self.language_actions.items():
            action.setChecked(code == language)
        self.apply_language()

    def apply_language(self):
        en = self.language == 'en'
        self.menu_button.setToolTip('Menu' if en else 'Меню')
        self.settings_menu.setTitle('Settings' if en else 'Настройки')
        self.language_menu.setTitle('Language' if en else 'Язык')
        self.background_menu.setTitle('Background' if en else 'Фон')
        self.default_background_action.setText('Default background' if en else 'Стандартный фон')
        self.choose_background_action.setText('Choose image…' if en else 'Выбрать изображение…')
        self.model_label.setText(self.model_path or ('No model selected' if en else 'Модель не выбрана'))
        self.choose_button.setText('Open GGUF model…' if en else 'Открыть GGUF-модель…')
        self.chat.setPlaceholderText('Select a supported GGUF model, then send a message.' if en else 'Выберите поддерживаемую GGUF-модель и отправьте сообщение.')
        self.input.setPlaceholderText('Message the local model…' if en else 'Напишите локальной модели…')
        self.send_button.setText('Send' if en else 'Отправить')
        self.tabs.setTabText(0, 'Chat' if en else 'Чат')
        self.tabs.setTabText(1, 'Console / Logs' if en else 'Консоль / Логи')
        self.log_view.setPlaceholderText('Diagnostics, errors and generation status appear here.' if en else 'Здесь отображаются диагностика, ошибки и состояние генерации.')
        self.temperature_label.setText('Temperature' if en else 'Температура')
        self.top_p_label.setText('Top-p' if en else 'Top-p')
        self.max_new_label.setText('Max new tokens' if en else 'Макс. новых токенов')
        self.stop_button.setText('Stop' if en else 'Стоп')
        self.new_button.setText('New chat' if en else 'Новый чат')
        if self.worker is None:
            self.status_label.setText('Ready · custom NumPy GGUF inference' if en else 'Готово · собственный движок NumPy GGUF')

    def choose_model(self):
        if self.worker is not None:
            QMessageBox.information(self, 'Generation running', 'Stop the current generation before switching models.')
            return
        start = str(Path(__file__).resolve().parents[1] / 'models')
        path, _ = QFileDialog.getOpenFileName(self, 'Select GGUF model', start, 'GGUF models (*.gguf)')
        if path:
            resolved = str(Path(path).resolve())
            if self.session is not None and self.session.path != resolved:
                self.session.close()
                self.session = None
            if self.model_path != resolved:
                self.metadata = None
            self.model_path = resolved
            self._temperature_user_set = False
            self._top_p_user_set = False
            self.temperature.blockSignals(True); self.temperature.setValue(.8); self.temperature.blockSignals(False)
            self.top_p.blockSignals(True); self.top_p.setValue(.95); self.top_p.blockSignals(False)
            self.model_label.setText(resolved)
            self.status_label.setText('Модель выбрана; она загрузится при следующей генерации.' if self.language == 'ru' else 'Model selected; it will be loaded for the next generation.')
            logging.getLogger(__name__).info('Selected model: %s', resolved)

    def new_chat(self):
        if self.worker is not None:
            self.stop_generation()
        self.messages.clear()
        self.chat.clear()
        self.input.clear()

    def stop_generation(self):
        self.stop_event.set()
        self.status_label.setText('Stopping after the current operation…')

    def send(self):
        content = self.input.text().strip()
        if not content:
            return
        if not self.model_path:
            QMessageBox.warning(self, 'No model', 'Choose a .gguf model first.')
            return
        if self.worker is not None:
            self.stop_generation()
            return

        self.messages.append({'role': 'user', 'content': content})
        self.chat.append(f'You: {content}\nLocalAI: ')
        self.input.clear()
        self.current_answer = ''
        self.send_button.setEnabled(False)
        self.stop_event = threading.Event()
        try:
            if self.metadata is None:
                with GGUFFile(self.model_path) as gguf:
                    self.metadata = gguf.metadata
            prompt_metadata = dict(self.metadata)
            prompt_metadata['_localai_language'] = self.language
            model_name = str(prompt_metadata.get('general.name', '')).lower()
            if 'saiga_llama3_8b' in model_name:
                if not self._temperature_user_set:
                    self.temperature.blockSignals(True); self.temperature.setValue(.75); self.temperature.blockSignals(False)
                if not self._top_p_user_set:
                    self.top_p.blockSignals(True); self.top_p.setValue(.92); self.top_p.blockSignals(False)
            prompt = format_chat(self.messages, prompt_metadata)
        except Exception as exc:
            self.messages.pop()
            self.send_button.setEnabled(True)
            QMessageBox.critical(self, 'GGUF error', str(exc))
            logging.getLogger(__name__).exception('Could not prepare prompt')
            return

        self.thread = QThread(self)
        settings = {'temperature': self.temperature.value(), 'top_p': self.top_p.value(), 'max_new': self.max_new.value(), 'language': self.language}
        self.worker = GenerationWorker(self.model_path, prompt, settings, self.stop_event, self.session)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.text.connect(self.append_token, Qt.ConnectionType.QueuedConnection)
        self.worker.status.connect(self.status_label.setText, Qt.ConnectionType.QueuedConnection)
        self.worker.error.connect(self.show_generation_error, Qt.ConnectionType.QueuedConnection)
        self.worker.session_ready.connect(self.set_session, Qt.ConnectionType.QueuedConnection)
        self.worker.done.connect(self.thread.quit)
        self.worker.done.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.generation_finished, Qt.ConnectionType.QueuedConnection)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @Slot(object)
    def set_session(self, session):
        self.session = session

    @Slot(str)
    def show_generation_error(self, message):
        QMessageBox.critical(self, 'Generation error', message)

    @Slot(str)
    def append_log(self, message):
        self.log_view.appendPlainText(message)

    @Slot(str)
    def append_token(self, text):
        self.current_answer += text
        self.chat.moveCursor(self.chat.textCursor().MoveOperation.End)
        self.chat.insertPlainText(text)
        self.chat.ensureCursorVisible()

    @Slot()
    def generation_finished(self):
        if self.messages:
            self.messages.append({'role': 'assistant', 'content': self.current_answer})
        self.worker = None
        self.thread = None
        self.send_button.setEnabled(True)

    def closeEvent(self, event):
        if self.worker is not None and self.thread is not None and self.thread.isRunning():
            self.stop_event.set()
            self.thread.quit()
            self.thread.wait()
        if self.session is not None:
            self.session.close()
            self.session = None
        logging.getLogger().removeHandler(self._log_handler)
        super().closeEvent(event)
