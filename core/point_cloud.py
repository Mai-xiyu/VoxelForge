"""
PointCloudProcessor — 点云处理模块

功能:
1. 加载多种点云格式 (PLY, PCD, XYZ, LAS, E57)
2. 3D 高斯 Splatting .ply 解析 (提取位置 + SH DC 颜色)
3. 统计离群点剔除
4. 法线估计
5. Poisson 表面重建 → 转为三角网格
6. 体素下采样
7. 点云 → 直接体素化 (无需重建)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PointCloudData:
    """点云数据容器"""
    points: np.ndarray                     # (N, 3) float64
    colors: Optional[np.ndarray] = None    # (N, 3) uint8
    normals: Optional[np.ndarray] = None   # (N, 3) float64
    metadata: Dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def has_colors(self) -> bool:
        return self.colors is not None and self.colors.shape[0] > 0

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)

    @property
    def extent(self) -> np.ndarray:
        vmin, vmax = self.bounds
        return vmax - vmin

    def summary(self) -> Dict:
        return {
            "points": self.n_points,
            "has_colors": self.has_colors,
            "has_normals": self.normals is not None,
            "extent": self.extent.tolist(),
        }


class PointCloudProcessor:
    """
    点云处理器

    Usage::

        pcp = PointCloudProcessor()
        pc = pcp.load("scan.ply")
        pc = pcp.remove_outliers(pc)
        pc = pcp.estimate_normals(pc)
        mesh = pcp.poisson_reconstruct(pc, depth=9)
    """

    def __init__(self) -> None:
        self._o3d = None
        try:
            import open3d as o3d
            self._o3d = o3d
            logger.info("Open3D %s loaded", o3d.__version__)
        except ImportError:
            logger.warning("Open3D not installed — point cloud functions will be limited")

        self._plyfile = None
        try:
            from plyfile import PlyData
            self._plyfile = PlyData
            logger.info("plyfile loaded for 3DGS parsing")
        except ImportError:
            pass

    # ── 加载 ────────────────────────────────────────────────────

    def load(self, path: str | Path) -> PointCloudData:
        """
        加载通用点云文件 (PLY, PCD, XYZ, LAS, E57)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Point cloud not found: {path}")

        ext = path.suffix.lower()

        if ext == ".ply":
            # 先尝试作为 3DGS ply 解析
            if self._is_3dgs_ply(path):
                logger.info("Detected 3DGS .ply format: %s", path.name)
                return self._load_3dgs_ply(path)

        if self._o3d is None:
            raise RuntimeError("Open3D is required for point cloud loading")

        logger.info("Loading point cloud: %s", path.name)
        pcd = self._o3d.io.read_point_cloud(str(path))

        points = np.asarray(pcd.points, dtype=np.float64)

        colors = None
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

        normals = None
        if pcd.has_normals():
            normals = np.asarray(pcd.normals, dtype=np.float64)

        result = PointCloudData(
            points=points,
            colors=colors,
            normals=normals,
            metadata={"format": ext, "source": str(path)},
        )

        logger.info("Loaded: %d points, colors=%s, normals=%s",
                     result.n_points, result.has_colors, normals is not None)
        return result

    # ── 3D Gaussian Splatting PLY ───────────────────────────────

    def _is_3dgs_ply(self, path: Path) -> bool:
        """检测 PLY 是否为 3DGS 格式（含 SH 球谐系数字段）"""
        if self._plyfile is None:
            return False
        try:
            ply = self._plyfile.read(str(path))
            props = [p.name for p in ply["vertex"].properties]
            return "f_dc_0" in props and "f_dc_1" in props and "f_dc_2" in props
        except Exception:
            return False

    def _load_3dgs_ply(self, path: Path) -> PointCloudData:
        """
        解析 3DGS .ply 文件。
        提取: x, y, z 位置 + f_dc_0/1/2 (SH DC 分量 → RGB)
        """
        if self._plyfile is None:
            raise RuntimeError("plyfile package required for 3DGS .ply loading")

        ply = self._plyfile.read(str(path))
        vertex = ply["vertex"]

        # 位置
        x = vertex["x"]
        y = vertex["y"]
        z = vertex["z"]
        points = np.column_stack([x, y, z]).astype(np.float64)

        # SH DC 分量 → RGB
        # 3DGS 的 f_dc 是球谐函数 0 阶系数，需要转换
        # C0 = 0.28209479177387814
        C0 = 0.28209479177387814
        f_dc_0 = vertex["f_dc_0"]
        f_dc_1 = vertex["f_dc_1"]
        f_dc_2 = vertex["f_dc_2"]

        r = np.clip((f_dc_0 * C0 + 0.5) * 255, 0, 255).astype(np.uint8)
        g = np.clip((f_dc_1 * C0 + 0.5) * 255, 0, 255).astype(np.uint8)
        b = np.clip((f_dc_2 * C0 + 0.5) * 255, 0, 255).astype(np.uint8)
        colors = np.column_stack([r, g, b])

        # 提取额外属性
        props = [p.name for p in vertex.properties]
        metadata = {
            "format": "3dgs_ply",
            "source": str(path),
            "properties": props,
        }

        # 如果有 opacity/scale 信息也记录
        if "opacity" in props:
            metadata["has_opacity"] = True
        if "scale_0" in props:
            metadata["has_scale"] = True

        result = PointCloudData(points=points, colors=colors, metadata=metadata)
        logger.info("3DGS PLY: %d gaussians loaded", result.n_points)
        return result

    # ── 离群点剔除 ──────────────────────────────────────────────

    def remove_outliers(
        self,
        pc: PointCloudData,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0,
    ) -> PointCloudData:
        """统计离群点剔除"""
        if self._o3d is None:
            logger.warning("Open3D unavailable, skipping outlier removal")
            return pc

        pcd = self._to_o3d(pc)
        pcd_clean, mask = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )

        mask_np = np.asarray(mask)
        removed = pc.n_points - mask_np.sum()
        logger.info("Outlier removal: removed %d / %d points", removed, pc.n_points)

        return PointCloudData(
            points=np.asarray(pcd_clean.points, dtype=np.float64),
            colors=pc.colors[mask_np] if pc.has_colors else None,
            normals=np.asarray(pcd_clean.normals, dtype=np.float64) if pcd_clean.has_normals() else None,
            metadata={**pc.metadata, "outliers_removed": int(removed)},
        )

    # ── 法线估计 ────────────────────────────────────────────────

    def estimate_normals(
        self,
        pc: PointCloudData,
        radius: float = 0.1,
        max_nn: int = 30,
    ) -> PointCloudData:
        """使用 Open3D 估计点法线"""
        if self._o3d is None:
            raise RuntimeError("Open3D required for normal estimation")

        pcd = self._to_o3d(pc)
        pcd.estimate_normals(
            search_param=self._o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn
            )
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

        logger.info("Normals estimated for %d points", pc.n_points)
        return PointCloudData(
            points=pc.points.copy(),
            colors=pc.colors,
            normals=np.asarray(pcd.normals, dtype=np.float64),
            metadata=pc.metadata,
        )

    # ── Poisson 表面重建 ────────────────────────────────────────

    def poisson_reconstruct(
        self,
        pc: PointCloudData,
        depth: int = 9,
        width: float = 0,
        scale: float = 1.1,
        linear_fit: bool = False,
    ) -> "MeshData":
        """
        Poisson 表面重建，将点云转换为三角网格。

        Returns MeshData (from mesh_processor)
        """
        from core.mesh_processor import MeshData

        if self._o3d is None:
            raise RuntimeError("Open3D required for Poisson reconstruction")

        if pc.normals is None:
            logger.info("No normals found, estimating...")
            pc = self.estimate_normals(pc)

        pcd = self._to_o3d(pc)

        logger.info("Running Poisson reconstruction (depth=%d)...", depth)
        mesh, densities = self._o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
        )

        # 移除低密度面（噪声）
        densities_np = np.asarray(densities)
        quantile = np.quantile(densities_np, 0.01)
        vertices_to_remove = densities_np < quantile
        mesh.remove_vertices_by_mask(vertices_to_remove)

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.int32)

        # 传递顶点颜色
        vertex_colors = None
        if mesh.has_vertex_colors():
            vertex_colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)

        normals = None
        if mesh.has_vertex_normals():
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

        result = MeshData(
            vertices=vertices,
            faces=faces,
            normals=normals,
            vertex_colors=vertex_colors,
            metadata={
                "reconstructed_from": "poisson",
                "depth": depth,
                "original_points": pc.n_points,
            },
        )

        logger.info("Poisson result: %d verts, %d faces", result.n_vertices, result.n_faces)
        return result

    # ── 体素下采样 ──────────────────────────────────────────────

    def voxel_downsample(self, pc: PointCloudData, voxel_size: float = 0.05) -> PointCloudData:
        """体素下采样，减少点数"""
        if self._o3d is None:
            raise RuntimeError("Open3D required for voxel downsampling")

        pcd = self._to_o3d(pc)
        down = pcd.voxel_down_sample(voxel_size=voxel_size)

        colors = None
        if down.has_colors():
            colors = (np.asarray(down.colors) * 255).astype(np.uint8)

        normals = None
        if down.has_normals():
            normals = np.asarray(down.normals, dtype=np.float64)

        result = PointCloudData(
            points=np.asarray(down.points, dtype=np.float64),
            colors=colors,
            normals=normals,
            metadata={**pc.metadata, "downsampled_voxel_size": voxel_size},
        )

        logger.info("Downsampled: %d → %d points (voxel=%.4f)",
                     pc.n_points, result.n_points, voxel_size)
        return result

    # ── 点云 → 直接体素化 ──────────────────────────────────────

    def direct_voxelize(
        self,
        pc: PointCloudData,
        target_height: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将点云直接体素化（无需先重建网格）。
        返回 (occupancy, color_grid)。
        """
        points = pc.points.copy()

        # 居中到正象限 + 缩放
        vmin = points.min(axis=0)
        vmax = points.max(axis=0)
        extent = vmax - vmin
        h = extent[1] if extent[1] > 1e-8 else extent.max()
        scale = target_height / h

        scaled = (points - vmin) * scale

        res_x = max(1, int(np.ceil(extent[0] * scale)))
        res_y = max(1, int(np.ceil(extent[1] * scale)))
        res_z = max(1, int(np.ceil(extent[2] * scale)))

        occupancy = np.zeros((res_x, res_y, res_z), dtype=np.int32)
        color_grid = np.zeros((res_x, res_y, res_z, 3), dtype=np.uint8)

        # 将点映射到体素坐标
        voxel_coords = np.floor(scaled).astype(int)
        voxel_coords[:, 0] = np.clip(voxel_coords[:, 0], 0, res_x - 1)
        voxel_coords[:, 1] = np.clip(voxel_coords[:, 1], 0, res_y - 1)
        voxel_coords[:, 2] = np.clip(voxel_coords[:, 2], 0, res_z - 1)

        for i in range(voxel_coords.shape[0]):
            x, y, z = voxel_coords[i]
            occupancy[x, y, z] = 1
            if pc.has_colors:
                color_grid[x, y, z] = pc.colors[i]
            else:
                color_grid[x, y, z] = [128, 128, 128]

        logger.info("Direct voxelized: %d points → %d×%d×%d grid, %d voxels",
                     pc.n_points, res_x, res_y, res_z, occupancy.sum())

        return occupancy, color_grid

    # ── Open3D 辅助 ─────────────────────────────────────────────

    def _to_o3d(self, pc: PointCloudData):
        """PointCloudData → Open3D PointCloud"""
        pcd = self._o3d.geometry.PointCloud()
        pcd.points = self._o3d.utility.Vector3dVector(pc.points)
        if pc.has_colors:
            pcd.colors = self._o3d.utility.Vector3dVector(pc.colors.astype(float) / 255.0)
        if pc.normals is not None:
            pcd.normals = self._o3d.utility.Vector3dVector(pc.normals)
        return pcd
