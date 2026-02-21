#!/usr/bin/env python3
"""
VoxelForge — 3D 数据 → Minecraft 方块转换器

入口点：初始化 GPU、i18n、加载配置、启动 GUI。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging() -> None:
    """配置日志系统"""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt="%H:%M:%S",
    )
    # 降低第三方库日志级别
    for lib in ("trimesh", "PIL", "matplotlib", "urllib3"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def load_config() -> dict:
    """加载配置文件"""
    import yaml

    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main() -> int:
    """主入口"""
    setup_logging()
    logger = logging.getLogger("VoxelForge")
    logger.info("Starting VoxelForge...")

    # 加载配置
    config = load_config()
    logger.info("Config loaded: %d sections", len(config))

    # 初始化 i18n
    from i18n import I18nManager
    i18n = I18nManager.instance()
    lang = config.get("language", "auto")

    i18n.load(lang if lang != "auto" else "auto")
    logger.info("i18n: %s (%d locales available)", i18n.current_locale, len(i18n.available_locales))

    # 初始化 GPU 管理器
    from core.gpu_manager import GpuManager

    compute_config = config.get("compute", {})
    gm = GpuManager(
        gpu_mem_limit_pct=compute_config.get("gpu_memory_limit_pct", 80),
        cpu_reserved_cores=compute_config.get("cpu_reserved_cores", 2),
    )
    gm.initialize()
    logger.info("GPU: backend=%s, device=%s", gm.backend, gm.device_info.device_name)

    # 启动 GUI
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont, QIcon

    app = QApplication(sys.argv)
    app.setApplicationName("VoxelForge")
    app.setApplicationVersion("0.1.0")

    # 设置全局窗口图标
    icon_path = PROJECT_ROOT / "gui" / "resources" / "icons" / "logo.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # 全局字体
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # 全局深色样式
    app.setStyleSheet("""
        QWidget {
            background: #1e1e2e;
            color: #cdd6f4;
        }
        QToolTip {
            background: #181825;
            color: #cdd6f4;
            border: 1px solid #313244;
            padding: 4px;
        }
        QScrollBar:vertical {
            background: #181825;
            width: 10px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #45475a;
            border-radius: 5px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background: #585b70;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }
        QScrollBar:horizontal {
            background: #181825;
            height: 10px;
        }
        QScrollBar::handle:horizontal {
            background: #45475a;
            border-radius: 5px;
            min-width: 20px;
        }
    """)

    from gui.main_window import MainWindow

    window = MainWindow(gm)

    # 加载调色板预览
    palette_path = PROJECT_ROOT / "config" / "block_palette.json"
    if palette_path.exists():
        window.load_palette_preview(str(palette_path))

    window.show()
    logger.info("VoxelForge GUI ready")

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
