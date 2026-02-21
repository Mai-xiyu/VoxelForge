"""
InfoPanelWidget — 基于 QWebEngineView 的信息面板

封装 QWebEngineView + QWebChannel，加载内嵌 HTML 信息页面。
如果 WebEngine 不可用则降级为 QLabel。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import QLabel, QWidget, QVBoxLayout

logger = logging.getLogger(__name__)

# HTML 文件所在目录
_HTML_DIR = Path(__file__).parent / "html"
_FALLBACK_HTML_PATH = Path(__file__).parent.parent / "web" / "index.html"


class InfoPanelWidget(QWidget):
    """
    封装 QWebEngineView 的信息面板。

    优先加载 web_panels/html/index.html，
    降级顺序: web_panels/html → gui/web/index.html → 内嵌 HTML → QLabel。
    """

    def __init__(self, i18n=None, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._i18n = i18n
        self._web_view = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        widget = self._create_web_view()
        layout.addWidget(widget)

    @property
    def web_view(self):
        """返回内部 QWebEngineView 实例（可能为 None）"""
        return self._web_view

    def _create_web_view(self) -> QWidget:
        """尝试创建 QWebEngineView，失败则降级"""
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            from PySide6.QtWebChannel import QWebChannel

            web = QWebEngineView()
            channel = QWebChannel()

            # 暴露 i18n 给 JS
            if self._i18n is not None:
                channel.registerObject("i18n", self._i18n)
            web.page().setWebChannel(channel)

            # 按优先级加载 HTML
            html_path = self._resolve_html_path()
            if html_path and html_path.exists():
                web.load(QUrl.fromLocalFile(str(html_path)))
            else:
                web.setHtml(self._fallback_html())

            self._web_view = web
            return web

        except ImportError:
            logger.warning("QWebEngineView not available, using fallback label")
            label = QLabel("Web 面板不可用\n请安装 PySide6-WebEngine")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background: #1e1e2e; color: #cdd6f4;")
            return label

    @staticmethod
    def _resolve_html_path() -> Optional[Path]:
        """按优先级查找 HTML 文件"""
        candidates = [
            _HTML_DIR / "index.html",
            _FALLBACK_HTML_PATH,
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    @staticmethod
    def _fallback_html() -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {
                    font-family: 'Segoe UI', sans-serif;
                    background: #1e1e2e; color: #cdd6f4;
                    display: flex; justify-content: center; align-items: center;
                    height: 100vh; margin: 0;
                }
                .container { text-align: center; }
                h1 { color: #89b4fa; }
                p { color: #a6adc8; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>⚒ VoxelForge</h1>
                <p>3D 数据 → Minecraft 方块转换器</p>
                <p>选择左侧管线并点击"开始转换"</p>
            </div>
        </body>
        </html>
        """

    def update_status(self, text: str) -> None:
        """通过 JS 更新状态栏"""
        if self._web_view:
            self._web_view.page().runJavaScript(
                f'if(typeof updateStatus==="function")updateStatus("{text}");'
            )

    def update_gpu_limit(self, pct: int) -> None:
        """通过 JS 更新 GPU 内存限制显示"""
        if self._web_view:
            self._web_view.page().runJavaScript(
                f'if(typeof updateGpuLimit==="function")updateGpuLimit({pct});'
            )
