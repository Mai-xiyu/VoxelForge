"""
MeshProcessor — 网格处理模块

功能:
1. 统一的网格数据结构 MeshData
2. 多格式导入 (OBJ, FBX, GLTF, PLY, STL, DXF)
3. 网格修复 (法线修正, 孔洞修复, 退化面移除)
4. 水密性检测与修复
5. 网格简化 (二次误差度量)
6. 实体化 (薄壁加厚)
7. 变换 (缩放, 旋转, 平移, 居中)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MeshData:
    """统一的网格数据容器"""
    vertices: np.ndarray           # (N, 3) float32
    faces: np.ndarray              # (M, 3) int32
    normals: Optional[np.ndarray] = None   # (N, 3) float32
    uvs: Optional[np.ndarray] = None       # (N, 2) float32
    vertex_colors: Optional[np.ndarray] = None  # (N, 3) uint8
    texture_path: Optional[str] = None
    source_path: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def extent(self) -> np.ndarray:
        vmin, vmax = self.bounds
        return vmax - vmin

    @property
    def center(self) -> np.ndarray:
        vmin, vmax = self.bounds
        return (vmin + vmax) / 2.0

    def summary(self) -> Dict:
        vmin, vmax = self.bounds
        return {
            "vertices": self.n_vertices,
            "faces": self.n_faces,
            "has_normals": self.normals is not None,
            "has_uvs": self.uvs is not None,
            "has_colors": self.vertex_colors is not None,
            "has_texture": self.texture_path is not None,
            "extent": self.extent.tolist(),
            "center": self.center.tolist(),
        }


# ── 已知可导入的扩展名 ────────────────────────────────────────
SUPPORTED_MESH_EXTENSIONS = {
    ".obj", ".fbx", ".gltf", ".glb", ".ply", ".stl",
    ".dxf", ".3ds", ".dae", ".off", ".3mf",
}


class MeshProcessor:
    """
    网格处理器 — 导入, 修复, 简化, 变换

    Usage::

        mp = MeshProcessor()
        mesh = mp.load("model.obj")
        mesh = mp.repair(mesh)
        mesh = mp.simplify(mesh, target_faces=50000)
        mesh = mp.center_and_scale(mesh, target_height=128)
    """

    def __init__(self) -> None:
        self._trimesh = None
        try:
            import trimesh
            self._trimesh = trimesh
            logger.info("trimesh %s loaded", trimesh.__version__)
        except ImportError:
            logger.warning("trimesh not installed — mesh import will be limited")

    # ── 导入 ────────────────────────────────────────────────────

    def load(self, path: str | Path) -> MeshData:
        """
        从文件加载网格。
        自动识别格式，提取顶点/面/法线/UV/顶点颜色/纹理路径。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Mesh file not found: {path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_MESH_EXTENSIONS:
            raise ValueError(f"Unsupported mesh format: {ext}")

        if self._trimesh is None:
            raise RuntimeError("trimesh is required for mesh loading")

        logger.info("Loading mesh: %s", path.name)

        scene_or_mesh = self._trimesh.load(str(path), process=False)

        # 处理 Scene 类型（可能包含多个 mesh）
        if isinstance(scene_or_mesh, self._trimesh.Scene):
            meshes = list(scene_or_mesh.geometry.values())
            if not meshes:
                raise ValueError(f"No geometry found in: {path}")
            # 合并所有子网格
            mesh = self._trimesh.util.concatenate(meshes)
            logger.info("Merged %d sub-meshes from scene", len(meshes))
        else:
            mesh = scene_or_mesh

        # 提取数据
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)

        normals = None
        if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None:
            normals = np.array(mesh.vertex_normals, dtype=np.float32)

        uvs = None
        if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            uvs = np.array(mesh.visual.uv, dtype=np.float32)

        vertex_colors = None
        texture_path = None

        if hasattr(mesh, "visual"):
            visual = mesh.visual
            # 顶点颜色
            if hasattr(visual, "vertex_colors") and visual.vertex_colors is not None:
                vc = np.array(visual.vertex_colors)
                if vc.shape[1] >= 3:
                    vertex_colors = vc[:, :3].astype(np.uint8)

            # 纹理
            if hasattr(visual, "material") and hasattr(visual.material, "image"):
                tex_img = visual.material.image
                if tex_img is not None:
                    # 保存纹理路径供后续使用
                    texture_path = str(path.with_suffix("")) + "_texture.png"
                    try:
                        tex_img.save(texture_path)
                    except Exception:
                        texture_path = None

        result = MeshData(
            vertices=vertices,
            faces=faces,
            normals=normals,
            uvs=uvs,
            vertex_colors=vertex_colors,
            texture_path=texture_path,
            source_path=str(path),
            metadata={
                "format": ext,
                "original_vertices": vertices.shape[0],
                "original_faces": faces.shape[0],
            },
        )

        logger.info("Loaded: %d verts, %d faces, uvs=%s, colors=%s",
                     result.n_vertices, result.n_faces,
                     uvs is not None, vertex_colors is not None)

        return result

    # ── 网格修复 ────────────────────────────────────────────────

    def repair(self, mesh: MeshData) -> MeshData:
        """
        修复网格：
        - 移除退化面 (面积为0)
        - 合并重复顶点
        - 法线一致化
        - 尝试填补孔洞
        """
        if self._trimesh is None:
            logger.warning("trimesh not available, skipping repair")
            return mesh

        tri = self._trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            process=False,
        )

        # 移除退化面
        mask = tri.area_faces > 0
        if not mask.all():
            removed = (~mask).sum()
            tri.update_faces(mask)
            logger.info("Removed %d degenerate faces", removed)

        # 合并重复顶点
        tri.merge_vertices()

        # 法线一致化
        tri.fix_normals()

        # 填补孔洞
        if not tri.is_watertight:
            try:
                self._trimesh.repair.fill_holes(tri)
                logger.info("Attempted hole filling. Watertight: %s", tri.is_watertight)
            except Exception as e:
                logger.warning("Hole filling failed: %s", e)

        result = MeshData(
            vertices=np.array(tri.vertices, dtype=np.float32),
            faces=np.array(tri.faces, dtype=np.int32),
            normals=np.array(tri.vertex_normals, dtype=np.float32),
            uvs=mesh.uvs,
            vertex_colors=mesh.vertex_colors,
            texture_path=mesh.texture_path,
            source_path=mesh.source_path,
            metadata={**mesh.metadata, "repaired": True},
        )

        logger.info("Repair done: %d verts, %d faces, watertight=%s",
                     result.n_vertices, result.n_faces, tri.is_watertight)
        return result

    # ── 水密性检测 ──────────────────────────────────────────────

    def check_watertight(self, mesh: MeshData) -> bool:
        """检测网格是否水密（闭合）"""
        if self._trimesh is None:
            return False
        tri = self._trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
        return tri.is_watertight

    # ── 网格简化 ────────────────────────────────────────────────

    def simplify(self, mesh: MeshData, target_faces: int) -> MeshData:
        """
        简化网格到目标面数 (二次误差度量)。
        如果当前面数已 <= target_faces 则跳过。
        """
        if mesh.n_faces <= target_faces:
            logger.info("Mesh already has %d faces (target=%d), skip simplification",
                        mesh.n_faces, target_faces)
            return mesh

        if self._trimesh is None:
            logger.warning("trimesh not available, skipping simplification")
            return mesh

        tri = self._trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)

        # 使用 trimesh 简化
        try:
            simplified = tri.simplify_quadric_decimation(target_faces)
            logger.info("Simplified: %d → %d faces", mesh.n_faces, len(simplified.faces))
        except Exception as e:
            logger.warning("Simplification failed: %s, returning original", e)
            return mesh

        result = MeshData(
            vertices=np.array(simplified.vertices, dtype=np.float32),
            faces=np.array(simplified.faces, dtype=np.int32),
            normals=np.array(simplified.vertex_normals, dtype=np.float32) if hasattr(simplified, "vertex_normals") else None,
            uvs=None,  # UV 在简化后通常无效
            vertex_colors=None,
            texture_path=mesh.texture_path,
            source_path=mesh.source_path,
            metadata={**mesh.metadata, "simplified_from": mesh.n_faces},
        )
        return result

    # ── 变换 ────────────────────────────────────────────────────

    def center_and_scale(self, mesh: MeshData, target_height: int = 128) -> MeshData:
        """
        居中网格并缩放，使 Y 轴高度 == target_height。
        不改变原始 mesh，返回新的 MeshData。
        """
        verts = mesh.vertices.copy()

        # 居中
        center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
        verts -= center

        # 缩放
        extent = verts.max(axis=0) - verts.min(axis=0)
        h = extent[1]
        if h < 1e-8:
            h = extent.max()
        scale = target_height / h

        verts *= scale

        # 平移到正象限 (最小坐标 = 0)
        vmin = verts.min(axis=0)
        verts -= vmin

        return MeshData(
            vertices=verts,
            faces=mesh.faces.copy(),
            normals=mesh.normals,
            uvs=mesh.uvs,
            vertex_colors=mesh.vertex_colors,
            texture_path=mesh.texture_path,
            source_path=mesh.source_path,
            metadata={**mesh.metadata, "scaled_height": target_height, "scale_factor": float(scale)},
        )

    def transform(
        self,
        mesh: MeshData,
        translate: Optional[Tuple[float, float, float]] = None,
        scale: Optional[float] = None,
        rotate_y_deg: Optional[float] = None,
    ) -> MeshData:
        """通用变换：平移 + 均匀缩放 + 绕 Y 轴旋转"""
        verts = mesh.vertices.copy()

        if rotate_y_deg is not None:
            rad = np.radians(rotate_y_deg)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            rot = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=np.float32)
            center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
            verts = (verts - center) @ rot.T + center

        if scale is not None:
            center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
            verts = (verts - center) * scale + center

        if translate is not None:
            verts += np.array(translate, dtype=np.float32)

        return MeshData(
            vertices=verts,
            faces=mesh.faces.copy(),
            normals=mesh.normals,
            uvs=mesh.uvs,
            vertex_colors=mesh.vertex_colors,
            texture_path=mesh.texture_path,
            source_path=mesh.source_path,
            metadata=mesh.metadata,
        )

    # ── 工具方法 ────────────────────────────────────────────────

    @staticmethod
    def supported_extensions() -> List[str]:
        return sorted(SUPPORTED_MESH_EXTENSIONS)
