"""LocalAI GUI entry point; logs are written to logs/localai.log."""
import logging
import sys
import traceback
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox
from app.window import MainWindow


def main():
    base = Path(__file__).resolve().parent
    log_dir = base / 'logs'
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        handlers=[logging.FileHandler(log_dir / 'localai.log', encoding='utf-8')],
        force=True,
    )
    app = QApplication(sys.argv)
    app.setApplicationName('LocalAI')
    try:
        window = MainWindow()
        window.show()
        return app.exec()
    except Exception:
        logging.getLogger(__name__).exception('Fatal application startup error')
        QMessageBox.critical(None, 'LocalAI startup error', traceback.format_exc())
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
