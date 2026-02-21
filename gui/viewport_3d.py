"""
Viewport3D — PyVista/VTK 3D 视口

嵌入 PySide6 的 3D 渲染视口，用于预览：
- 导入的网格
- 体素化结果
- 点云
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Viewport3D:
    """
    3D 预览视口 (PyVista + Qt)

    在 PySide6 窗口中嵌入 VTK 渲染窗口。
    提供网格预览、体素预览、点云预览功能。

    Usage::

        viewport = Viewport3D()
        widget = viewport.get_widget(parent)
        viewport.show_mesh(vertices, faces)
        viewport.show_voxels(occupancy, colors)
    """

    def __init__(self) -> None:
        self._pv = None
        self._plotter = None
        self._widget = None

        try:
            import pyvista as pv
            import pyvistaqt
            self._pv = pv
            self._pvqt = pyvistaqt
            # 设置 PyVista 主题
            pv.global_theme.anti_aliasing = "msaa"
            pv.global_theme.background = "#1e1e2e"
            logger.info("PyVista %s loaded for 3D viewport", pv.__version__)
        except ImportError:
            logger.warning("PyVista/pyvistaqt not installed — 3D viewport disabled")

    @property
    def available(self) -> bool:
        return self._pv is not None

    def get_widget(self, parent=None):
        """创建并返回 Qt widget"""
        if not self.available:
            # Fallback: 返回一个简单的 QLabel
            from PySide6.QtWidgets import QLabel
            from PySide6.QtCore import Qt
            label = QLabel("3D 视口不可用\n请安装 pyvista 和 pyvistaqt", parent)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background: #1e1e2e; color: #cdd6f4; font-size: 14px;")
            return label

        self._plotter = self._pvqt.QtInteractor(parent)
        self._plotter.set_background("#1e1e2e")
        self._plotter.add_axes()
        self._widget = self._plotter
        return self._plotter

    def clear(self) -> None:
        """清除所有 actor"""
        if self._plotter:
            self._plotter.clear()
            self._plotter.add_axes()

    def show_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_colors: Optional[np.ndarray] = None,
        opacity: float = 1.0,
    ) -> None:
        """显示三角网格"""
        if not self._plotter:
            return

        pv = self._pv
        n_faces = faces.shape[0]

        # PyVista faces 格式: [3, v0, v1, v2, 3, v0, v1, v2, ...]
        pv_faces = np.column_stack([
            np.full(n_faces, 3, dtype=np.int32),
            faces,
        ]).flatten()

        mesh = pv.PolyData(vertices, pv_faces)

        kwargs = {"opacity": opacity, "smooth_shading": True}

        if vertex_colors is not None and vertex_colors.shape[0] == vertices.shape[0]:
            mesh["colors"] = vertex_colors
            kwargs["scalars"] = "colors"
            kwargs["rgb"] = True

        self._plotter.add_mesh(mesh, **kwargs)
        self._plotter.reset_camera()

    def show_voxels(
        self,
        occupancy: np.ndarray,
        color_grid: Optional[np.ndarray] = None,
        max_display: int = 200000,
    ) -> None:
        """
        显示体素网格为小方块。
        如果体素太多会自动降采样显示。
        """
        if not self._plotter:
            return

        pv = self._pv
        coords = np.argwhere(occupancy > 0).astype(np.float32)

        if coords.shape[0] == 0:
            logger.warning("No voxels to display")
            return

        # 如果太多体素，随机采样
        if coords.shape[0] > max_display:
            indices = np.random.choice(coords.shape[0], max_display, replace=False)
            coords = coords[indices]
            logger.info("Downsampled voxel display: %d / %d", max_display, occupancy.sum())

        # 创建点云可视化
        cloud = pv.PolyData(coords)

        if color_grid is not None:
            colors = np.zeros((coords.shape[0], 3), dtype=np.uint8)
            for i in range(coords.shape[0]):
                x, y, z = int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])
                colors[i] = color_grid[x, y, z]
            cloud["colors"] = colors
            glyphs = cloud.glyph(geom=pv.Cube(x_length=0.9, y_length=0.9, z_length=0.9))
            self._plotter.add_mesh(glyphs, scalars="colors", rgb=True)
        else:
            glyphs = cloud.glyph(geom=pv.Cube(x_length=0.9, y_length=0.9, z_length=0.9))
            self._plotter.add_mesh(glyphs, color="#89b4fa")

        self._plotter.reset_camera()

    def show_point_cloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        point_size: float = 2.0,
        max_display: int = 500000,
    ) -> None:
        """显示点云"""
        if not self._plotter:
            return

        pv = self._pv

        if points.shape[0] > max_display:
            indices = np.random.choice(points.shape[0], max_display, replace=False)
            points = points[indices]
            if colors is not None:
                colors = colors[indices]

        cloud = pv.PolyData(points.astype(np.float32))

        if colors is not None:
            cloud["colors"] = colors
            self._plotter.add_mesh(
                cloud,
                scalars="colors",
                rgb=True,
                point_size=point_size,
                render_points_as_spheres=True,
            )
        else:
            self._plotter.add_mesh(
                cloud,
                color="#a6e3a1",
                point_size=point_size,
                render_points_as_spheres=True,
            )

        self._plotter.reset_camera()

    def screenshot(self, path: str) -> None:
        """保存当前视图截图"""
        if self._plotter:
            self._plotter.screenshot(path)

    def set_camera(
        self, position: Tuple[float, float, float] = None,
        focal_point: Tuple[float, float, float] = None,
    ) -> None:
        """设置相机位置"""
        if self._plotter:
            if position:
                self._plotter.camera_position = position
            if focal_point:
                self._plotter.camera.focal_point = focal_point
