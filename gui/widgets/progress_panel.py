"""
ProgressPanel — 进度显示面板
"""

from __future__ import annotations

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QVBoxLayout, QWidget,
)


class ProgressPanel(QFrame):
    """进度显示面板"""
    cancel_requested = Signal()

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            ProgressPanel {
                background: #1e1e2e;
                border: 1px solid #313244;
                border-radius: 8px;
                padding: 12px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._status_label = QLabel("就绪")
        self._status_label.setStyleSheet("color: #cdd6f4; font-size: 13px;")
        layout.addWidget(self._status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #45475a;
                border-radius: 4px;
                text-align: center;
                color: #cdd6f4;
                background: #181825;
                height: 22px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #89b4fa, stop:1 #74c7ec);
                border-radius: 3px;
            }
        """)
        layout.addWidget(self._progress_bar)

        btn_row = QHBoxLayout()
        self._detail_label = QLabel("")
        self._detail_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        btn_row.addWidget(self._detail_label, 1)

        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.setStyleSheet("""
            QPushButton {
                background: #f38ba8; color: #1e1e2e;
                border: none; border-radius: 4px;
                padding: 4px 16px; font-weight: bold;
            }
            QPushButton:hover { background: #eba0ac; }
        """)
        self._cancel_btn.clicked.connect(self.cancel_requested.emit)
        self._cancel_btn.setVisible(False)
        btn_row.addWidget(self._cancel_btn)

        layout.addLayout(btn_row)

    @Slot(float, str)
    def update_progress(self, percent: float, message: str) -> None:
        self._progress_bar.setValue(int(percent))
        self._status_label.setText(message)
        self._cancel_btn.setVisible(0 < percent < 100)

    def set_detail(self, text: str) -> None:
        self._detail_label.setText(text)

    def reset(self) -> None:
        self._progress_bar.setValue(0)
        self._status_label.setText("就绪")
        self._detail_label.setText("")
        self._cancel_btn.setVisible(False)
