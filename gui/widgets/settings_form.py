"""
SettingsForm — 通用设置表单控件
"""

from __future__ import annotations

from typing import Dict, List

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from i18n import I18nManager


class SettingsForm(QFrame):
    """通用设置表单控件"""
    value_changed = Signal(str, object)

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(6)
        self._widgets: Dict[str, QWidget] = {}
        self.setStyleSheet("""
            QLabel { color: #cdd6f4; font-size: 12px; }
            QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
                background: #181825; color: #cdd6f4;
                border: 1px solid #45475a; border-radius: 4px;
                padding: 4px 8px;
            }
            QCheckBox { color: #cdd6f4; }
            QCheckBox::indicator { width: 16px; height: 16px; }
        """)

    def add_spin(self, key: str, label: str, min_v: int, max_v: int, default: int) -> QSpinBox:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(120)
        row.addWidget(lbl)
        spin = QSpinBox()
        spin.setRange(min_v, max_v)
        spin.setValue(default)
        spin.valueChanged.connect(lambda v: self.value_changed.emit(key, v))
        row.addWidget(spin)
        self._layout.addLayout(row)
        self._widgets[key] = spin
        return spin

    def add_double_spin(self, key: str, label: str, min_v: float, max_v: float,
                        default: float, step: float = 0.1, decimals: int = 2) -> QDoubleSpinBox:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(120)
        row.addWidget(lbl)
        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setValue(default)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.valueChanged.connect(lambda v: self.value_changed.emit(key, v))
        row.addWidget(spin)
        self._layout.addLayout(row)
        self._widgets[key] = spin
        return spin

    def add_combo(self, key: str, label: str, options: List[str], default: int = 0) -> QComboBox:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(120)
        row.addWidget(lbl)
        combo = QComboBox()
        combo.addItems(options)
        combo.setCurrentIndex(default)
        combo.currentTextChanged.connect(lambda v: self.value_changed.emit(key, v))
        row.addWidget(combo)
        self._layout.addLayout(row)
        self._widgets[key] = combo
        return combo

    def add_check(self, key: str, label: str, default: bool = False) -> QCheckBox:
        check = QCheckBox(label)
        check.setChecked(default)
        check.toggled.connect(lambda v: self.value_changed.emit(key, v))
        self._layout.addWidget(check)
        self._widgets[key] = check
        return check

    def add_file_picker(self, key: str, label: str, filter_str: str = "All Files (*)") -> QLineEdit:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(120)
        row.addWidget(lbl)
        edit = QLineEdit()
        edit.setPlaceholderText(I18nManager().t("settings.select_file"))
        row.addWidget(edit)
        btn = QPushButton("…")
        btn.setFixedWidth(32)
        btn.clicked.connect(lambda: self._pick_file(key, edit, filter_str))
        row.addWidget(btn)
        self._layout.addLayout(row)
        self._widgets[key] = edit
        return edit

    def _pick_file(self, key: str, edit: QLineEdit, filter_str: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, I18nManager().t("settings.select_file_title"), "", filter_str)
        if path:
            edit.setText(path)
            self.value_changed.emit(key, path)

    def get_value(self, key: str):
        w = self._widgets.get(key)
        if isinstance(w, (QSpinBox, QDoubleSpinBox)):
            return w.value()
        elif isinstance(w, QComboBox):
            return w.currentText()
        elif isinstance(w, QCheckBox):
            return w.isChecked()
        elif isinstance(w, QLineEdit):
            return w.text()
        return None

    def get_all_values(self) -> Dict:
        return {k: self.get_value(k) for k in self._widgets}
