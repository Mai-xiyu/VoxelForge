"""
Pipeline A — 3D 模型体素化管线

流程:
  导入 3D 模型 → 网格修复 → 缩放 → 体素化 → 颜色映射 → 稀疏网格 → 导出

支持输入: OBJ, FBX, GLTF, GLB, PLY, STL, DAE, 3DS, DXF
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image

from core.block_mapper import BlockMapper, MapperConfig
from core.gpu_manager import GpuManager
from core.mesh_processor import MeshData, MeshProcessor
from core.sparse_voxels import SparseVoxelGrid
from core.voxelizer import VoxelMode, Voxelizer, VoxelizeResult

logger = logging.getLogger(__name__)


@dataclass
class ModelPipelineConfig:
    """模型管线配置"""
    input_path: str = ""
    output_path: str = ""
    output_format: str = "litematic"     # litematic | schematic | mca
    target_height: int = 128
    voxel_mode: VoxelMode = VoxelMode.SOLID
    repair_mesh: bool = True
    simplify_mesh: bool = False
    simplify_target: int = 100000
    dithering: bool = True
    dither_strength: float = 1.0
    color_space: str = "lab"
    palette_path: str = "config/block_palette.json"
    category_whitelist: Optional[List[str]] = None
    category_blacklist: Optional[List[str]] = None
    mca_base_y: int = 64


@dataclass
class PipelineResult:
    """管线执行结果"""
    success: bool = True
    output_path: str = ""
    total_blocks: int = 0
    grid_size: tuple = (0, 0, 0)
    elapsed_sec: float = 0.0
    palette_used: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PipelineModel:
    """
    3D 模型 → Minecraft 转换管线

    Usage::

        gpu = GpuManager()
        gpu.initialize()
        pipeline = PipelineModel(gpu)
        result = pipeline.run(ModelPipelineConfig(
            input_path="model.obj",
            output_path="output.litematic",
            target_height=128,
        ))
    """

    def __init__(self, gpu_manager: GpuManager) -> None:
        self.gm = gpu_manager
        self.mesh_processor = MeshProcessor()
        self.voxelizer = Voxelizer(gpu_manager)
        self.block_mapper = BlockMapper()
        # 中间结果（可被 GUI 读取预览）
        self.current_mesh: Optional[MeshData] = None
        self.current_voxels: Optional[VoxelizeResult] = None
        self.current_grid: Optional[SparseVoxelGrid] = None

    def run(
        self,
        config: ModelPipelineConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> PipelineResult:
        """执行完整管线"""
        t0 = time.perf_counter()
        result = PipelineResult()

        def report(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        try:
            # ── Step 1: 导入 ──
            report(0.0, "导入 3D 模型…")
            mesh = self.mesh_processor.load(config.input_path)
            logger.info("Loaded: %d verts, %d faces", mesh.n_vertices, mesh.n_faces)

            # ── Step 2: 修复 ──
            if config.repair_mesh:
                report(10.0, "修复网格…")
                mesh = self.mesh_processor.repair(mesh)

            # ── Step 3: 简化 ──
            if config.simplify_mesh:
                report(15.0, "简化网格…")
                mesh = self.mesh_processor.simplify(mesh, config.simplify_target)

            # ── Step 4: 缩放 ──
            report(20.0, "缩放到目标高度…")
            mesh = self.mesh_processor.center_and_scale(mesh, config.target_height)
            self.current_mesh = mesh

            # ── Step 5: 体素化 ──
            report(25.0, "体素化中…")

            # 加载纹理
            texture = None
            if mesh.texture_path:
                try:
                    texture = Image.open(mesh.texture_path)
                except Exception as e:
                    result.warnings.append(f"Failed to load texture: {e}")

            voxel_result = self.voxelizer.voxelize(
                vertices=mesh.vertices,
                faces=mesh.faces,
                target_height=config.target_height,
                mode=config.voxel_mode,
                uvs=mesh.uvs,
                texture=texture,
                vertex_colors=mesh.vertex_colors,
                progress_callback=lambda p, m: report(25.0 + p * 0.3, m),
            )
            self.current_voxels = voxel_result
            result.grid_size = voxel_result.resolution

            # ── Step 6: 方块映射 ──
            report(60.0, "颜色映射…")
            self.block_mapper.load_palette(config.palette_path)

            mapper_config = MapperConfig(
                dithering=config.dithering,
                dither_strength=config.dither_strength,
                color_space=config.color_space,
                category_whitelist=set(config.category_whitelist) if config.category_whitelist else None,
                category_blacklist=set(config.category_blacklist) if config.category_blacklist else None,
            )

            block_grid = self.block_mapper.map_colors(
                voxel_result.color_grid,
                voxel_result.grid,
                mapper_config,
            )

            result.palette_used = self.block_mapper.active_size

            # ── Step 7: 构建稀疏网格 ──
            report(75.0, "构建稀疏体素网格…")
            sparse = SparseVoxelGrid()
            rx, ry, rz = voxel_result.resolution

            for x in range(rx):
                for y in range(ry):
                    for z in range(rz):
                        bid = block_grid[x, y, z]
                        if bid and bid != "":
                            sparse.set(x, y, z, bid)

            self.current_grid = sparse
            result.total_blocks = sparse.count

            # ── Step 8: 导出 ──
            report(85.0, "导出文件…")
            output_path = Path(config.output_path)

            if config.output_format == "litematic":
                from io_formats.litematic_exporter import LitematicExporter
                exporter = LitematicExporter()
                exporter.export(sparse, output_path, progress_callback=lambda p, m: report(85 + p * 0.14, m))

            elif config.output_format == "schematic":
                from io_formats.schematic_exporter import SchematicExporter
                exporter = SchematicExporter()
                exporter.export(sparse, output_path, progress_callback=lambda p, m: report(85 + p * 0.14, m))

            elif config.output_format == "mca":
                from io_formats.mca_exporter import McaExporter
                exporter = McaExporter()
                output_dir = output_path if output_path.is_dir() else output_path.parent / "region"
                files = exporter.export(sparse, output_dir, base_y=config.mca_base_y,
                                        progress_callback=lambda p, m: report(85 + p * 0.14, m))
                result.output_path = str(output_dir)
            else:
                raise ValueError(f"Unknown output format: {config.output_format}")

            if not result.output_path:
                result.output_path = str(output_path)

            result.success = True
            elapsed = time.perf_counter() - t0
            result.elapsed_sec = elapsed

            report(100.0, f"完成! ({elapsed:.1f}s, {result.total_blocks} 方块)")
            logger.info("Pipeline complete: %d blocks in %.1fs", result.total_blocks, elapsed)

        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            logger.error("Pipeline failed: %s", e, exc_info=True)
            report(100.0, f"错误: {e}")

        return result
