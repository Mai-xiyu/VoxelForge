"""
PaletteViewer — 方块调色板预览列表
"""

from __future__ import annotations

from typing import Dict, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QLabel, QScrollArea, QVBoxLayout, QWidget,
)


class PaletteViewer(QScrollArea):
    """方块调色板预览列表"""

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet("background: #1e1e2e; border: none;")

        self._container = QWidget()
        self._layout = QGridLayout(self._container)
        self._layout.setSpacing(4)
        self.setWidget(self._container)

    def load_palette(self, blocks: List[Dict]) -> None:
        """加载并显示调色板方块列表"""
        # 清空
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        cols = 6
        for i, block in enumerate(blocks):
            row, col = divmod(i, cols)

            frame = QFrame()
            frame.setFixedSize(60, 72)
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(2, 2, 2, 2)
            fl.setSpacing(2)

            # 颜色色块
            color_label = QLabel()
            color_label.setFixedSize(48, 36)
            r, g, b = block.get("rgb", [128, 128, 128])
            color_label.setStyleSheet(
                f"background: rgb({r},{g},{b}); border: 1px solid #45475a; border-radius: 4px;"
            )
            fl.addWidget(color_label, alignment=Qt.AlignCenter)

            # 名称
            name = block.get("id", "").split(":")[-1].replace("_", " ")
            text = QLabel(name[:8])
            text.setStyleSheet("color: #a6adc8; font-size: 9px;")
            text.setAlignment(Qt.AlignCenter)
            fl.addWidget(text)

            self._layout.addWidget(frame, row, col)
