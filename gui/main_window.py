"""
MainWindow — VoxelForge 主窗口

布局:
    ┌─────────────────────────────────────────────────┐
    │  菜单栏 + 工具栏                                  │
    ├────────────┬───────────────────────┬─────────────┤
    │  左侧面板   │    3D 视口 / Web面板    │  右侧面板    │
    │  (向导/设置)│   (PySide + PyVista)   │  (调色板等)  │
    ├────────────┴───────────────────────┴─────────────┤
    │  底部: 进度条 + 日志                               │
    └─────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal, Slot, QUrl
from PySide6.QtGui import QAction, QIcon, QKeySequence, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QTabWidget, QFileDialog, QMessageBox, QStatusBar,
    QDockWidget, QToolBar, QLabel, QStackedWidget, QStyle,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QLineEdit,
)

from core.gpu_manager import GpuManager
from gui.viewport_3d import Viewport3D
from gui.widgets import (
    LogViewer, PaletteViewer, ProgressPanel,
    QLogHandler, SettingsForm,
)
from i18n import I18nManager

logger = logging.getLogger(__name__)


# ── Pipeline Worker Thread ──────────────────────────────────

class PipelineWorker(QThread):
    """在后台线程中执行管线"""
    progress = Signal(float, str)
    finished = Signal(object)

    def __init__(self, pipeline, config) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.config = config

    def run(self) -> None:
        result = self.pipeline.run(
            self.config,
            progress_callback=self.progress.emit,
        )
        self.finished.emit(result)


class MainWindow(QMainWindow):
    """VoxelForge 主窗口"""

    def __init__(self, gpu_manager: GpuManager) -> None:
        super().__init__()
        self.gm = gpu_manager
        self.i18n = I18nManager()
        self._worker: Optional[PipelineWorker] = None

        # 设置窗口图标
        icon_path = Path(__file__).parent / "resources" / "icons" / "logo.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self._init_ui()
        self._init_menus()
        self._init_toolbar()
        self._init_logging()
        self._update_title()

        # 连接语言切换信号
        self.i18n.language_changed.connect(self._on_language_changed)

    def _init_ui(self) -> None:
        """初始化界面布局"""
        self.setMinimumSize(1280, 800)

        # 中央 widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── 主分割器 (左 | 中 | 右) ──
        self._main_splitter = QSplitter(Qt.Horizontal)

        # 左侧面板 — 向导 + 设置
        self._left_panel = self._build_left_panel()
        self._main_splitter.addWidget(self._left_panel)

        # 中间 — 3D 视口 + Web 面板 (Tab)
        self._center_tabs = QTabWidget()
        self._center_tabs.setStyleSheet("""
            QTabWidget::pane { border: none; }
            QTabBar::tab {
                background: #181825; color: #a6adc8;
                padding: 8px 16px; border: none;
                border-bottom: 2px solid transparent;
            }
            QTabBar::tab:selected {
                color: #89b4fa;
                border-bottom: 2px solid #89b4fa;
            }
        """)

        # 3D 视口 tab
        self._viewport = Viewport3D()
        viewport_widget = self._viewport.get_widget(self._center_tabs)
        self._center_tabs.addTab(viewport_widget, self.i18n.t("gui.tab_3d_viewport"))

        # Web 面板 tab (内嵌 QWebEngineView)
        self._web_panel = self._build_web_panel()
        self._center_tabs.addTab(self._web_panel, self.i18n.t("gui.tab_info_panel"))

        self._main_splitter.addWidget(self._center_tabs)

        # 右侧面板 — 调色板 + 统计
        self._right_panel = self._build_right_panel()
        self._main_splitter.addWidget(self._right_panel)

        # 设置分割比例
        self._main_splitter.setSizes([280, 720, 240])

        main_layout.addWidget(self._main_splitter, 1)

        # ── 底部面板 (进度 + 日志) ──
        self._bottom_splitter = QSplitter(Qt.Vertical)

        self._progress_panel = ProgressPanel()
        self._progress_panel.cancel_requested.connect(self._on_cancel_pipeline)
        self._bottom_splitter.addWidget(self._progress_panel)

        self._log_viewer = LogViewer()
        self._log_viewer.setMaximumHeight(150)
        self._bottom_splitter.addWidget(self._log_viewer)

        main_layout.addWidget(self._bottom_splitter, 0)

        # 状态栏
        self._status_bar = QStatusBar()
        self._status_bar.setStyleSheet("background: #181825; color: #a6adc8;")
        self.setStatusBar(self._status_bar)

        # 全局样式
        self.setStyleSheet("""
            QMainWindow { background: #1e1e2e; }
            QSplitter::handle { background: #313244; width: 2px; }
        """)

    def _build_left_panel(self) -> QWidget:
        """构建左侧设置面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        panel.setStyleSheet("background: #181825; border-radius: 8px;")
        panel.setMinimumWidth(260)
        panel.setMaximumWidth(400)

        # 标题
        title = QLabel("⚒ VoxelForge")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #89b4fa; padding: 8px;")
        layout.addWidget(title)

        # 管线选择
        self._pipeline_tabs = QTabWidget()
        self._pipeline_tabs.setStyleSheet("""
            QTabBar::tab {
                background: #1e1e2e; color: #a6adc8;
                padding: 6px 10px; font-size: 11px;
                border: none; border-bottom: 2px solid transparent;
            }
            QTabBar::tab:selected { color: #a6e3a1; border-bottom-color: #a6e3a1; }
        """)

        # 管线 A: 3D 模型
        self._model_settings = self._build_model_settings()
        self._pipeline_tabs.addTab(self._model_settings, self.i18n.t("gui.pipeline_model"))

        # 管线 B: 城市/GIS
        self._city_settings = self._build_city_settings()
        self._pipeline_tabs.addTab(self._city_settings, self.i18n.t("gui.pipeline_city"))

        # 管线 C: 点云/扫描
        self._scan_settings = self._build_scan_settings()
        self._pipeline_tabs.addTab(self._scan_settings, self.i18n.t("gui.pipeline_scan"))

        layout.addWidget(self._pipeline_tabs, 1)

        # 执行按钮
        from PySide6.QtWidgets import QPushButton
        self._run_btn = QPushButton(f"▶ {self.i18n.t('gui.start_convert')}")
        self._run_btn.setStyleSheet("""
            QPushButton {
                background: #a6e3a1; color: #1e1e2e;
                border: none; border-radius: 8px;
                padding: 12px; font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background: #94e2d5; }
            QPushButton:disabled { background: #45475a; color: #6c7086; }
        """)
        self._run_btn.clicked.connect(self._on_run_pipeline)
        layout.addWidget(self._run_btn)

        return panel

    def _build_model_settings(self) -> SettingsForm:
        """3D 模型管线设置"""
        t = self.i18n.t
        form = SettingsForm()
        form.add_file_picker("input_path", t("settings.input_file"),
                            "3D Models (*.obj *.fbx *.gltf *.glb *.ply *.stl *.dxf)")
        form.add_spin("target_height", t("settings.target_height"), 16, 384, 128)
        form.add_combo("voxel_mode", t("settings.voxel_mode"),
                       [t("settings.voxel_solid"), t("settings.voxel_surface")], 0)
        form.add_check("repair_mesh", t("settings.repair_mesh"), True)
        form.add_check("simplify_mesh", t("settings.simplify_mesh"), False)
        form.add_spin("simplify_target", t("settings.simplify_target"), 1000, 500000, 100000)
        form.add_check("dithering", t("settings.dithering"), True)
        form.add_double_spin("dither_strength", t("settings.dither_strength"), 0.0, 2.0, 1.0)
        form.add_combo("color_space", t("settings.color_space"), ["CIE L*a*b*", "RGB"], 0)
        form.add_combo("output_format", t("settings.output_format"), ["litematic", "schematic", "mca"], 0)
        return form

    def _build_city_settings(self) -> SettingsForm:
        """城市管线设置"""
        t = self.i18n.t
        form = SettingsForm()
        form.add_file_picker("heightmap_path", t("settings.heightmap"), "Images (*.png *.tiff *.jpg *.bmp)")
        form.add_file_picker("buildings_path", t("settings.buildings_data"), "GeoJSON (*.geojson *.json)")
        form.add_spin("max_height", t("settings.max_height"), 16, 320, 128)
        form.add_double_spin("scale", t("settings.scale"), 0.01, 100.0, 1.0)
        form.add_double_spin("water_level", t("settings.water_level"), 0.0, 1.0, 0.15, 0.01)
        form.add_combo("output_format", t("settings.output_format"), ["litematic", "schematic", "mca"], 0)
        return form

    def _build_scan_settings(self) -> SettingsForm:
        """扫描管线设置"""
        t = self.i18n.t
        form = SettingsForm()
        form.add_file_picker("input_path", t("settings.pointcloud_file"),
                            "Point Clouds (*.ply *.pcd *.xyz *.las *.e57)")
        form.add_spin("target_height", t("settings.target_height"), 16, 384, 128)
        form.add_check("remove_outliers", t("settings.remove_outliers"), True)
        form.add_check("use_reconstruction", t("settings.poisson_recon"), False)
        form.add_spin("poisson_depth", t("settings.poisson_depth"), 4, 12, 9)
        form.add_check("dithering", t("settings.dithering"), True)
        form.add_combo("output_format", t("settings.output_format"), ["litematic", "schematic", "mca"], 0)
        return form

    def _build_right_panel(self) -> QWidget:
        """构建右侧信息面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        panel.setStyleSheet("background: #181825; border-radius: 8px;")
        panel.setMinimumWidth(220)
        panel.setMaximumWidth(360)

        # GPU 信息
        t = self.i18n.t
        self._gpu_title_label = QLabel(t("gui.gpu_info"))
        self._gpu_title_label.setStyleSheet("color: #89b4fa; font-size: 13px; font-weight: bold;")
        layout.addWidget(self._gpu_title_label)

        self._gpu_info = QLabel(f"{t('gui.gpu_backend')}: {self.gm.backend}\n{t('gui.gpu_device')}: {self.gm.device_info.device_name}")
        self._gpu_info.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 4px;")
        layout.addWidget(self._gpu_info)

        # 调色板预览
        self._palette_title_label = QLabel(t("gui.block_palette"))
        self._palette_title_label.setStyleSheet("color: #89b4fa; font-size: 13px; font-weight: bold; margin-top: 12px;")
        layout.addWidget(self._palette_title_label)

        self._palette_viewer = PaletteViewer()
        layout.addWidget(self._palette_viewer, 1)

        # 统计
        self._stats_title_label = QLabel(t("gui.statistics"))
        self._stats_title_label.setStyleSheet("color: #89b4fa; font-size: 13px; font-weight: bold; margin-top: 12px;")
        layout.addWidget(self._stats_title_label)

        self._stats_label = QLabel(t("gui.no_data"))
        self._stats_label.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 4px;")
        self._stats_label.setWordWrap(True)
        layout.addWidget(self._stats_label)

        return panel

    def _build_web_panel(self) -> QWidget:
        """构建 Web 面板 (QWebEngineView 或 fallback)"""
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            from PySide6.QtWebChannel import QWebChannel

            web = QWebEngineView()
            channel = QWebChannel()
            # 暴露 i18n 给 JS
            channel.registerObject("i18n", self.i18n)
            web.page().setWebChannel(channel)

            # 加载内嵌 HTML
            html_path = Path(__file__).parent / "web" / "index.html"
            if html_path.exists():
                web.load(QUrl.fromLocalFile(str(html_path)))
            else:
                web.setHtml(self._fallback_html())

            self._web_view = web
            return web

        except ImportError:
            logger.warning("QWebEngineView not available, using fallback")
            label = QLabel(self.i18n.t("gui.web_panel_unavailable"))
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background: #1e1e2e; color: #cdd6f4;")
            self._web_view = None
            return label

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

    # ── 菜单栏 ──────────────────────────────────────────────────

    def _init_menus(self) -> None:
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar { background: #181825; color: #cdd6f4; }
            QMenuBar::item:selected { background: #313244; }
            QMenu { background: #1e1e2e; color: #cdd6f4; border: 1px solid #313244; }
            QMenu::item:selected { background: #313244; }
        """)

        t = self.i18n.t

        # 文件菜单
        self._file_menu = menubar.addMenu(t("menu.file._title") + "(&F)")

        self._open_act = QAction(t("menu.file.open") + "(&O)", self)
        self._open_act.setShortcut(QKeySequence.Open)
        self._open_act.triggered.connect(self._on_open_file)
        self._file_menu.addAction(self._open_act)

        self._file_menu.addSeparator()

        self._exit_act = QAction(t("menu.file.exit") + "(&Q)", self)
        self._exit_act.setShortcut(QKeySequence.Quit)
        self._exit_act.triggered.connect(self.close)
        self._file_menu.addAction(self._exit_act)

        # 视图菜单
        self._view_menu = menubar.addMenu(t("menu.view._title") + "(&V)")

        self._reset_cam_act = QAction(t("viewport.reset_camera"), self)
        self._reset_cam_act.triggered.connect(lambda: self._viewport._plotter.reset_camera() if self._viewport._plotter else None)
        self._view_menu.addAction(self._reset_cam_act)

        # 工具菜单
        self._tools_menu = menubar.addMenu(t("menu.tools._title") + "(&T)")

        self._gpu_diag_act = QAction(t("gui.toolbar_gpu_diag"), self)
        self._gpu_diag_act.triggered.connect(self._on_gpu_info)
        self._tools_menu.addAction(self._gpu_diag_act)

        # 语言菜单
        self._lang_menu = menubar.addMenu(t("menu.view.language") + "(&L)")
        for locale_id, meta in self.i18n.available_locales.items():
            act = QAction(f"{meta.get('flag', '')} {meta.get('native', locale_id)}", self)
            act.triggered.connect(lambda checked, lid=locale_id: self.i18n.switch(lid))
            self._lang_menu.addAction(act)

        # 帮助菜单
        self._help_menu = menubar.addMenu(t("menu.help._title") + "(&H)")
        self._about_act = QAction(t("menu.help.about"), self)
        self._about_act.triggered.connect(self._on_about)
        self._help_menu.addAction(self._about_act)

    def _init_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏")
        toolbar.setStyleSheet("""
            QToolBar { background: #181825; border: none; spacing: 4px; }
            QToolButton { color: #cdd6f4; padding: 6px; }
            QToolButton:hover { background: #313244; border-radius: 4px; }
        """)
        toolbar.setMovable(False)

        style = self.style()

        t = self.i18n.t

        self._tb_open = QAction(style.standardIcon(QStyle.SP_DialogOpenButton), t("gui.toolbar_open"), self)
        self._tb_open.triggered.connect(self._on_open_file)
        toolbar.addAction(self._tb_open)

        toolbar.addSeparator()

        self._tb_run = QAction(style.standardIcon(QStyle.SP_MediaPlay), t("gui.toolbar_run"), self)
        self._tb_run.triggered.connect(self._on_run_pipeline)
        toolbar.addAction(self._tb_run)

        toolbar.addSeparator()

        self._tb_save = QAction(style.standardIcon(QStyle.SP_DialogSaveButton), t("gui.toolbar_save"), self)
        self._tb_save.triggered.connect(lambda: None)  # placeholder
        toolbar.addAction(self._tb_save)

        self._tb_info = QAction(style.standardIcon(QStyle.SP_MessageBoxInformation), t("gui.toolbar_gpu_diag"), self)
        self._tb_info.triggered.connect(self._on_gpu_info)
        toolbar.addAction(self._tb_info)

        self.addToolBar(toolbar)

    def _init_logging(self) -> None:
        """设置日志输出到 UI"""
        handler = QLogHandler(self._log_viewer)
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    def _update_title(self) -> None:
        backend = self.gm.backend if self.gm else "N/A"
        self.setWindowTitle(self.i18n.t("gui.window_title_fmt", backend=backend))

    # ── 事件处理 ────────────────────────────────────────────────

    def _on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, self.i18n.t("gui.open_file"), "",
            "All Supported (*.obj *.fbx *.gltf *.glb *.ply *.stl *.dxf *.png *.tiff *.litematic);;"
            "3D Models (*.obj *.fbx *.gltf *.glb *.ply *.stl *.dxf);;"
            "Point Clouds (*.ply *.pcd *.xyz *.las *.e57);;"
            "Height Maps (*.png *.tiff *.jpg *.bmp);;"
            "Litematic (*.litematic)"
        )
        if not path:
            return

        ext = Path(path).suffix.lower()
        if ext in (".obj", ".fbx", ".gltf", ".glb", ".stl", ".dxf", ".dae"):
            self._pipeline_tabs.setCurrentIndex(0)
            self._model_settings._widgets["input_path"].setText(path)
            self._preview_mesh(path)
        elif ext == ".ply":
            # 可能是网格也可能是点云
            self._pipeline_tabs.setCurrentIndex(2)
            self._scan_settings._widgets["input_path"].setText(path)
        elif ext in (".png", ".tiff", ".jpg", ".bmp"):
            self._pipeline_tabs.setCurrentIndex(1)
            self._city_settings._widgets["heightmap_path"].setText(path)
        elif ext == ".litematic":
            self._on_import_litematic(path)

    def _preview_mesh(self, path: str) -> None:
        """预览导入的网格"""
        try:
            from core.mesh_processor import MeshProcessor
            mp = MeshProcessor()
            mesh = mp.load(path)
            self._viewport.clear()
            self._viewport.show_mesh(mesh.vertices, mesh.faces, mesh.vertex_colors)
            t = self.i18n.t
            self._stats_label.setText(
                f"{t('gui.vertices')}: {mesh.n_vertices:,}\n"
                f"{t('gui.faces')}: {mesh.n_faces:,}\n"
                f"{t('gui.dimensions')}: {mesh.extent[0]:.1f} × {mesh.extent[1]:.1f} × {mesh.extent[2]:.1f}\n"
                f"UV: {'✓' if mesh.uvs is not None else '✗'}\n"
                f"{t('gui.vertex_colors')}: {'✓' if mesh.vertex_colors is not None else '✗'}"
            )
        except Exception as e:
            logger.error("Preview failed: %s", e)

    def _on_import_litematic(self, path: str) -> None:
        """导入 .litematic 文件"""
        try:
            from io_formats.importers import LitematicImporter
            importer = LitematicImporter()
            grid = importer.load(path)
            logger.info("Imported .litematic: %d blocks", grid.count)
        except Exception as e:
            QMessageBox.warning(self, self.i18n.t("gui.import_error"), str(e))

    def _on_run_pipeline(self) -> None:
        """执行当前选中的管线"""
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, self.i18n.t("common.warning"), self.i18n.t("gui.task_running"))
            return

        tab_index = self._pipeline_tabs.currentIndex()
        self._run_btn.setEnabled(False)

        try:
            if tab_index == 0:
                self._run_model_pipeline()
            elif tab_index == 1:
                self._run_city_pipeline()
            elif tab_index == 2:
                self._run_scan_pipeline()
        except Exception as e:
            logger.error("Failed to start pipeline: %s", e)
            self._run_btn.setEnabled(True)

    def _run_model_pipeline(self) -> None:
        from pipelines.pipeline_model import PipelineModel, ModelPipelineConfig
        from core.voxelizer import VoxelMode

        vals = self._model_settings.get_all_values()
        if not vals.get("input_path"):
            QMessageBox.warning(self, self.i18n.t("common.warning"), self.i18n.t("gui.select_input"))
            self._run_btn.setEnabled(True)
            return

        # 输出路径
        output_format = vals.get("output_format", "litematic")
        ext_map = {"litematic": ".litematic", "schematic": ".schem", "mca": ""}
        out_ext = ext_map.get(output_format, ".litematic")

        if out_ext:
            output_path, _ = QFileDialog.getSaveFileName(
                self, self.i18n.t("gui.save_as"), f"output{out_ext}", f"*{out_ext}")
        else:
            output_path = QFileDialog.getExistingDirectory(self, self.i18n.t("gui.select_output_dir"))

        if not output_path:
            self._run_btn.setEnabled(True)
            return

        # Combo index 0 = solid, 1 = surface (regardless of locale)
        voxel_combo = self._model_settings._widgets.get("voxel_mode")
        voxel_mode = VoxelMode.SOLID if (voxel_combo and voxel_combo.currentIndex() == 0) else VoxelMode.SURFACE

        config = ModelPipelineConfig(
            input_path=vals["input_path"],
            output_path=output_path,
            output_format=output_format,
            target_height=vals.get("target_height", 128),
            voxel_mode=voxel_mode,
            repair_mesh=vals.get("repair_mesh", True),
            simplify_mesh=vals.get("simplify_mesh", False),
            simplify_target=vals.get("simplify_target", 100000),
            dithering=vals.get("dithering", True),
            dither_strength=vals.get("dither_strength", 1.0),
            color_space="lab" if "lab" in vals.get("color_space", "").lower() else "rgb",
            palette_path="config/block_palette.json",
        )

        pipeline = PipelineModel(self.gm)
        self._start_worker(pipeline, config)

    def _run_city_pipeline(self) -> None:
        from pipelines.pipeline_city import PipelineCity, CityPipelineConfig

        vals = self._city_settings.get_all_values()
        if not vals.get("heightmap_path") and not vals.get("buildings_path"):
            QMessageBox.warning(self, self.i18n.t("common.warning"), self.i18n.t("gui.select_heightmap"))
            self._run_btn.setEnabled(True)
            return

        output_format = vals.get("output_format", "litematic")
        ext_map = {"litematic": ".litematic", "schematic": ".schem", "mca": ""}
        out_ext = ext_map.get(output_format, ".litematic")

        if out_ext:
            output_path, _ = QFileDialog.getSaveFileName(self, self.i18n.t("gui.save_as"), f"city{out_ext}", f"*{out_ext}")
        else:
            output_path = QFileDialog.getExistingDirectory(self, self.i18n.t("gui.select_output_dir"))

        if not output_path:
            self._run_btn.setEnabled(True)
            return

        config = CityPipelineConfig(
            heightmap_path=vals.get("heightmap_path", ""),
            buildings_path=vals.get("buildings_path", ""),
            output_path=output_path,
            output_format=output_format,
            max_height=vals.get("max_height", 128),
            scale=vals.get("scale", 1.0),
            water_level=vals.get("water_level", 0.15),
        )

        pipeline = PipelineCity()
        self._start_worker(pipeline, config)

    def _run_scan_pipeline(self) -> None:
        from pipelines.pipeline_scan import PipelineScan, ScanPipelineConfig

        vals = self._scan_settings.get_all_values()
        if not vals.get("input_path"):
            QMessageBox.warning(self, self.i18n.t("common.warning"), self.i18n.t("gui.select_pointcloud"))
            self._run_btn.setEnabled(True)
            return

        output_format = vals.get("output_format", "litematic")
        ext_map = {"litematic": ".litematic", "schematic": ".schem", "mca": ""}
        out_ext = ext_map.get(output_format, ".litematic")

        if out_ext:
            output_path, _ = QFileDialog.getSaveFileName(self, self.i18n.t("gui.save_as"), f"scan{out_ext}", f"*{out_ext}")
        else:
            output_path = QFileDialog.getExistingDirectory(self, self.i18n.t("gui.select_output_dir"))

        if not output_path:
            self._run_btn.setEnabled(True)
            return

        config = ScanPipelineConfig(
            input_path=vals["input_path"],
            output_path=output_path,
            output_format=output_format,
            target_height=vals.get("target_height", 128),
            remove_outliers=vals.get("remove_outliers", True),
            use_reconstruction=vals.get("use_reconstruction", False),
            poisson_depth=vals.get("poisson_depth", 9),
            dithering=vals.get("dithering", True),
            palette_path="config/block_palette.json",
        )

        pipeline = PipelineScan(self.gm)
        self._start_worker(pipeline, config)

    def _start_worker(self, pipeline, config) -> None:
        """启动管线工作线程"""
        self._progress_panel.reset()
        self._worker = PipelineWorker(pipeline, config)
        self._worker.progress.connect(self._progress_panel.update_progress)
        self._worker.finished.connect(self._on_pipeline_finished)
        self._worker.start()

    @Slot(object)
    def _on_pipeline_finished(self, result) -> None:
        """管线完成回调"""
        self._run_btn.setEnabled(True)

        t = self.i18n.t
        if hasattr(result, "success") and result.success:
            total = getattr(result, "total_blocks", 0)
            elapsed = getattr(result, "elapsed_sec", 0)
            self._stats_label.setText(
                f"✅ {t('gui.convert_success')}\n"
                f"{t('gui.blocks_count')}: {total:,}\n"
                f"{t('gui.time_elapsed')}: {elapsed:.1f}s\n"
                f"{t('gui.output')}: {getattr(result, 'output_path', 'N/A')}"
            )
            QMessageBox.information(self, t("gui.done_title"),
                t("gui.done_msg", blocks=f"{total:,}", elapsed=f"{elapsed:.1f}"))
        else:
            errors = getattr(result, "errors", [])
            err_text = "\n".join(errors) if errors else t("common.unknown")
            self._stats_label.setText(f"❌ {t('gui.convert_failed')}\n{err_text}")
            QMessageBox.critical(self, t("gui.error_title"),
                t("gui.error_msg", errors=err_text))

    def _on_cancel_pipeline(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(3000)
            self._run_btn.setEnabled(True)
            logger.warning(self.i18n.t("gui.pipeline_cancelled"))

    def _on_gpu_info(self) -> None:
        t = self.i18n.t
        snap = self.gm.snapshot()
        throttle = t("common.yes") if self.gm.should_throttle() else t("common.no")
        QMessageBox.information(self, t("gui.toolbar_gpu_diag"), (
            f"{t('gui.gpu_backend')}: {self.gm.backend}\n"
            f"{t('gui.gpu_device')}: {self.gm.device_info.device_name}\n"
            f"{t('gpu.gpu_mem')}: {snap.gpu_mem_used_mb:.0f} / {snap.gpu_mem_total_mb:.0f} MB\n"
            f"CPU: {snap.cpu_percent:.0f}%\n"
            f"RAM: {snap.ram_percent:.0f}%\n"
            f"{t('gpu.throttle_warning').split('⚠️')[-1].strip() if self.gm.should_throttle() else ''}"
        ))

    def _on_about(self) -> None:
        t = self.i18n.t
        QMessageBox.about(self, t("about.title"), t("gui.about_text"))

    @Slot(str)
    def _on_language_changed(self, locale_id: str) -> None:
        """语言切换后更新 UI 所有文本"""
        t = self.i18n.t

        # 窗口标题
        self._update_title()

        # 开始转换按钮
        self._run_btn.setText(f"▶ {t('gui.start_convert')}")

        # 中间 Tab 标签
        self._center_tabs.setTabText(0, t("gui.tab_3d_viewport"))
        self._center_tabs.setTabText(1, t("gui.tab_info_panel"))

        # 管线 Tab 标签
        self._pipeline_tabs.setTabText(0, t("gui.pipeline_model"))
        self._pipeline_tabs.setTabText(1, t("gui.pipeline_city"))
        self._pipeline_tabs.setTabText(2, t("gui.pipeline_scan"))

        # 右侧面板标签
        self._gpu_title_label.setText(t("gui.gpu_info"))
        self._gpu_info.setText(f"{t('gui.gpu_backend')}: {self.gm.backend}\n{t('gui.gpu_device')}: {self.gm.device_info.device_name}")
        self._palette_title_label.setText(t("gui.block_palette"))
        self._stats_title_label.setText(t("gui.statistics"))

        # 菜单栏
        self._file_menu.setTitle(t("menu.file._title") + "(&F)")
        self._open_act.setText(t("menu.file.open") + "(&O)")
        self._exit_act.setText(t("menu.file.exit") + "(&Q)")
        self._view_menu.setTitle(t("menu.view._title") + "(&V)")
        self._reset_cam_act.setText(t("viewport.reset_camera"))
        self._tools_menu.setTitle(t("menu.tools._title") + "(&T)")
        self._gpu_diag_act.setText(t("gui.toolbar_gpu_diag"))
        self._lang_menu.setTitle(t("menu.view.language") + "(&L)")
        self._help_menu.setTitle(t("menu.help._title") + "(&H)")
        self._about_act.setText(t("menu.help.about"))

        # 工具栏
        self._tb_open.setText(t("gui.toolbar_open"))
        self._tb_run.setText(t("gui.toolbar_run"))
        self._tb_save.setText(t("gui.toolbar_save"))
        self._tb_info.setText(t("gui.toolbar_gpu_diag"))

        # 重建设置表单（保留当前值）
        self._rebuild_settings_forms()

        logger.info("Language switched to: %s", locale_id)

    def _rebuild_settings_forms(self) -> None:
        """重建三个设置表单，保留用户已输入的值"""
        t = self.i18n.t
        current_idx = self._pipeline_tabs.currentIndex()

        # 保存旧值
        saved_model = self._model_settings.get_all_values()
        saved_city = self._city_settings.get_all_values()
        saved_scan = self._scan_settings.get_all_values()

        # 移除所有旧 tab（从后往前避免索引偏移）
        for i in range(2, -1, -1):
            self._pipeline_tabs.removeTab(i)

        # 构建新表单
        self._model_settings = self._build_model_settings()
        self._city_settings = self._build_city_settings()
        self._scan_settings = self._build_scan_settings()

        # 恢复值
        for form, saved in [
            (self._model_settings, saved_model),
            (self._city_settings, saved_city),
            (self._scan_settings, saved_scan),
        ]:
            for key, val in saved.items():
                w = form._widgets.get(key)
                if w is None:
                    continue
                if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                    w.setValue(val)
                elif isinstance(w, QComboBox):
                    idx_c = w.findText(str(val))
                    if idx_c >= 0:
                        w.setCurrentIndex(idx_c)
                elif isinstance(w, QCheckBox):
                    w.setChecked(bool(val))
                elif isinstance(w, QLineEdit):
                    w.setText(str(val) if val else "")

        # 插入新 tab
        self._pipeline_tabs.addTab(self._model_settings, t("gui.pipeline_model"))
        self._pipeline_tabs.addTab(self._city_settings, t("gui.pipeline_city"))
        self._pipeline_tabs.addTab(self._scan_settings, t("gui.pipeline_scan"))

        # 恢复选中的 tab
        self._pipeline_tabs.setCurrentIndex(current_idx)

    def load_palette_preview(self, palette_path: str = "config/block_palette.json") -> None:
        """加载并显示调色板预览"""
        try:
            from core.block_mapper import BlockMapper
            mapper = BlockMapper()
            mapper.load_palette(palette_path)
            preview = mapper.preview_palette(60)
            self._palette_viewer.load_palette(preview)
        except Exception as e:
            logger.warning("Failed to load palette: %s", e)
