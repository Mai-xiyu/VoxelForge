"""
Pipeline C — 扫描数据 / 点云 / 3DGS 管线

流程:
  导入点云/3DGS .ply
  → 离群点剔除
  → (可选) Poisson 表面重建 → 体素化
  → 或直接体素化
  → 颜色映射
  → 导出

支持输入: PLY (含3DGS), PCD, XYZ, LAS, E57
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from core.block_mapper import BlockMapper, MapperConfig
from core.gpu_manager import GpuManager
from core.mesh_processor import MeshProcessor
from core.point_cloud import PointCloudData, PointCloudProcessor
from core.sparse_voxels import SparseVoxelGrid
from core.voxelizer import VoxelMode, Voxelizer

logger = logging.getLogger(__name__)


@dataclass
class ScanPipelineConfig:
    """扫描管线配置"""
    input_path: str = ""
    output_path: str = ""
    output_format: str = "litematic"
    target_height: int = 128
    # 点云处理
    remove_outliers: bool = True
    outlier_neighbors: int = 20
    outlier_std: float = 2.0
    downsample_voxel_size: float = 0.0   # 0 = 不下采样
    # 重建方式
    use_reconstruction: bool = False      # True = Poisson 重建后体素化
    poisson_depth: int = 9
    # 颜色映射
    dithering: bool = True
    dither_strength: float = 1.0
    color_space: str = "lab"
    palette_path: str = "config/block_palette.json"
    category_whitelist: Optional[List[str]] = None
    category_blacklist: Optional[List[str]] = None
    # MCA
    mca_base_y: int = 64


@dataclass
class ScanPipelineResult:
    """扫描管线结果"""
    success: bool = True
    output_path: str = ""
    total_blocks: int = 0
    input_points: int = 0
    processed_points: int = 0
    grid_size: Tuple[int, int, int] = (0, 0, 0)
    is_3dgs: bool = False
    elapsed_sec: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PipelineScan:
    """
    点云 / 3DGS → Minecraft 转换管线

    Usage::

        gpu = GpuManager()
        gpu.initialize()
        pipeline = PipelineScan(gpu)
        result = pipeline.run(ScanPipelineConfig(
            input_path="scan.ply",
            output_path="scan_build.litematic",
        ))
    """

    def __init__(self, gpu_manager: GpuManager) -> None:
        self.gm = gpu_manager
        self.pc_processor = PointCloudProcessor()
        self.mesh_processor = MeshProcessor()
        self.voxelizer = Voxelizer(gpu_manager)
        self.block_mapper = BlockMapper()
        # 中间结果
        self.current_pc: Optional[PointCloudData] = None
        self.current_grid: Optional[SparseVoxelGrid] = None

    def run(
        self,
        config: ScanPipelineConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> ScanPipelineResult:
        """执行扫描管线"""
        t0 = time.perf_counter()
        result = ScanPipelineResult()

        def report(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        try:
            # ── Step 1: 导入点云 ──
            report(0.0, "导入点云…")
            pc = self.pc_processor.load(config.input_path)
            result.input_points = pc.n_points
            result.is_3dgs = pc.metadata.get("format") == "3dgs_ply"
            self.current_pc = pc

            logger.info("Point cloud: %d points, 3DGS=%s", pc.n_points, result.is_3dgs)

            # ── Step 2: 离群点剔除 ──
            if config.remove_outliers:
                report(10.0, "剔除离群点…")
                pc = self.pc_processor.remove_outliers(
                    pc,
                    nb_neighbors=config.outlier_neighbors,
                    std_ratio=config.outlier_std,
                )

            # ── Step 3: 下采样 ──
            if config.downsample_voxel_size > 0:
                report(20.0, "体素下采样…")
                pc = self.pc_processor.voxel_downsample(pc, config.downsample_voxel_size)

            result.processed_points = pc.n_points
            self.current_pc = pc

            # ── Step 4: 体素化 ──
            if config.use_reconstruction:
                # Poisson 重建 → 网格 → 体素化
                report(30.0, "Poisson 表面重建…")
                mesh = self.pc_processor.poisson_reconstruct(pc, depth=config.poisson_depth)

                report(50.0, "体素化重建网格…")
                voxel_result = self.voxelizer.voxelize(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    target_height=config.target_height,
                    mode=VoxelMode.SOLID,
                    vertex_colors=mesh.vertex_colors,
                    progress_callback=lambda p, m: report(50 + p * 0.15, m),
                )
                occupancy = voxel_result.grid
                color_grid = voxel_result.color_grid
                result.grid_size = voxel_result.resolution
            else:
                # 直接体素化
                report(30.0, "直接体素化…")
                occupancy, color_grid = self.pc_processor.direct_voxelize(
                    pc, target_height=config.target_height
                )
                result.grid_size = tuple(occupancy.shape)

            report(65.0, f"体素化完成: {occupancy.sum()} 个体素")

            # ── Step 5: 方块映射 ──
            report(70.0, "颜色映射…")
            self.block_mapper.load_palette(config.palette_path)

            mapper_config = MapperConfig(
                dithering=config.dithering,
                dither_strength=config.dither_strength,
                color_space=config.color_space,
                category_whitelist=set(config.category_whitelist) if config.category_whitelist else None,
                category_blacklist=set(config.category_blacklist) if config.category_blacklist else None,
            )

            block_grid = self.block_mapper.map_colors(color_grid, occupancy, mapper_config)

            # ── Step 6: 构建稀疏网格 ──
            report(80.0, "构建稀疏网格…")
            sparse = SparseVoxelGrid()
            rx, ry, rz = result.grid_size

            for x in range(rx):
                for y in range(ry):
                    for z in range(rz):
                        bid = block_grid[x, y, z]
                        if bid and bid != "":
                            sparse.set(x, y, z, bid)

            self.current_grid = sparse
            result.total_blocks = sparse.count

            # ── Step 7: 导出 ──
            report(85.0, "导出…")
            output_path = Path(config.output_path)

            if config.output_format == "litematic":
                from io_formats.litematic_exporter import LitematicExporter
                LitematicExporter().export(
                    sparse, output_path,
                    name="Scan Import",
                    progress_callback=lambda p, m: report(85 + p * 0.14, m),
                )
            elif config.output_format == "schematic":
                from io_formats.schematic_exporter import SchematicExporter
                SchematicExporter().export(
                    sparse, output_path,
                    progress_callback=lambda p, m: report(85 + p * 0.14, m),
                )
            elif config.output_format == "mca":
                from io_formats.mca_exporter import McaExporter
                out_dir = output_path if output_path.is_dir() else output_path.parent / "region"
                McaExporter().export(
                    sparse, out_dir, base_y=config.mca_base_y,
                    progress_callback=lambda p, m: report(85 + p * 0.14, m),
                )
                result.output_path = str(out_dir)

            if not result.output_path:
                result.output_path = str(output_path)

            result.success = True
            elapsed = time.perf_counter() - t0
            result.elapsed_sec = elapsed

            report(100.0, f"完成! ({elapsed:.1f}s, {result.total_blocks} 方块)")
            logger.info("Scan pipeline: %d points → %d blocks in %.1fs",
                        result.input_points, result.total_blocks, elapsed)

        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            logger.error("Scan pipeline failed: %s", e, exc_info=True)
            report(100.0, f"错误: {e}")

        return result
