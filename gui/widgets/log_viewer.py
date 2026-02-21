"""
LogViewer + QLogHandler — 日志查看窗口与 logging 桥接
"""

from __future__ import annotations

import logging

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QTextEdit, QWidget


class LogViewer(QTextEdit):
    """日志查看窗口"""

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 10))
        self.setStyleSheet("""
            QTextEdit {
                background: #11111b;
                color: #cdd6f4;
                border: 1px solid #313244;
                border-radius: 4px;
            }
        """)
        self._max_lines = 5000

    def append_log(self, level: str, message: str) -> None:
        color_map = {
            "DEBUG": "#a6adc8",
            "INFO": "#a6e3a1",
            "WARNING": "#f9e2af",
            "ERROR": "#f38ba8",
            "CRITICAL": "#f38ba8",
        }
        color = color_map.get(level, "#cdd6f4")
        self.append(f'<span style="color:{color}">[{level}] {message}</span>')

        # 限制行数
        if self.document().blockCount() > self._max_lines:
            cursor = self.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.movePosition(cursor.MoveOperation.Down, cursor.MoveMode.KeepAnchor, 500)
            cursor.removeSelectedText()


class QLogHandler(logging.Handler):
    """将 Python logging 输出到 LogViewer"""

    def __init__(self, log_viewer: LogViewer) -> None:
        super().__init__()
        self._viewer = log_viewer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._viewer.append_log(record.levelname, msg)
        except Exception:
            pass
