"""
Pipeline B — GIS / 城市数据导入管线

流程:
  导入 GIS 高度图 / 建筑 shapefile / OSM 数据
  → 解析地形 + 建筑 footprint
  → 高度拉伸生成体素
  → 方块映射 (地面/建筑/道路)
  → 导出

支持输入:
- 高度图 (PNG/TIFF 灰度图)
- GeoJSON / Shapefile 建筑数据
- OSM .pbf (简化)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from core.block_mapper import BlockMapper, MapperConfig
from core.sparse_voxels import SparseVoxelGrid

logger = logging.getLogger(__name__)


@dataclass
class CityPipelineConfig:
    """城市管线配置"""
    heightmap_path: str = ""          # 高度图路径(灰度图)
    buildings_path: str = ""          # 建筑数据路径 (GeoJSON/Shapefile)
    output_path: str = ""
    output_format: str = "litematic"
    max_height: int = 128             # MC 世界最大高度（方块数）
    terrain_block: str = "minecraft:grass_block"
    underground_block: str = "minecraft:stone"
    building_block: str = "minecraft:white_concrete"
    road_block: str = "minecraft:gray_concrete"
    water_block: str = "minecraft:water"
    water_level: float = 0.15         # 水面高度 (占比 0..1)
    scale: float = 1.0                # 导入缩放
    palette_path: str = "config/block_palette.json"
    mca_base_y: int = 64


@dataclass
class CityPipelineResult:
    """城市管线结果"""
    success: bool = True
    output_path: str = ""
    total_blocks: int = 0
    terrain_size: Tuple[int, int] = (0, 0)
    max_elevation: int = 0
    elapsed_sec: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PipelineCity:
    """
    GIS / 城市数据 → Minecraft 转换管线

    Usage::

        pipeline = PipelineCity()
        result = pipeline.run(CityPipelineConfig(
            heightmap_path="terrain.png",
            output_path="city.litematic",
        ))
    """

    def __init__(self) -> None:
        self.current_grid: Optional[SparseVoxelGrid] = None

    def run(
        self,
        config: CityPipelineConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> CityPipelineResult:
        """执行城市管线"""
        t0 = time.perf_counter()
        result = CityPipelineResult()

        def report(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        try:
            grid = SparseVoxelGrid()

            # ── Step 1: 导入高度图 ──
            if config.heightmap_path:
                report(0.0, "导入高度图…")
                grid = self._process_heightmap(config, grid, report)
            else:
                result.warnings.append("No heightmap provided, generating flat terrain")
                # 生成一个 64×64 的平坦地形
                for x in range(64):
                    for z in range(64):
                        grid.set(x, 0, z, config.terrain_block)

            result.terrain_size = self._get_xz_size(grid)
            result.max_elevation = self._get_max_y(grid)

            # ── Step 2: 导入建筑数据 ──
            if config.buildings_path:
                report(50.0, "导入建筑数据…")
                grid = self._process_buildings(config, grid, report)

            self.current_grid = grid
            result.total_blocks = grid.count

            # ── Step 3: 导出 ──
            report(80.0, "导出…")
            output_path = Path(config.output_path)

            if config.output_format == "litematic":
                from io_formats.litematic_exporter import LitematicExporter
                exporter = LitematicExporter()
                exporter.export(grid, output_path, name="City Import",
                               progress_callback=lambda p, m: report(80 + p * 0.19, m))
            elif config.output_format == "schematic":
                from io_formats.schematic_exporter import SchematicExporter
                SchematicExporter().export(grid, output_path,
                                          progress_callback=lambda p, m: report(80 + p * 0.19, m))
            elif config.output_format == "mca":
                from io_formats.mca_exporter import McaExporter
                out_dir = output_path if output_path.is_dir() else output_path.parent / "region"
                McaExporter().export(grid, out_dir, base_y=config.mca_base_y,
                                    progress_callback=lambda p, m: report(80 + p * 0.19, m))
                result.output_path = str(out_dir)

            if not result.output_path:
                result.output_path = str(output_path)

            result.success = True
            elapsed = time.perf_counter() - t0
            result.elapsed_sec = elapsed

            report(100.0, f"完成! ({elapsed:.1f}s, {result.total_blocks} 方块)")

        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            logger.error("City pipeline failed: %s", e, exc_info=True)

        return result

    def _process_heightmap(
        self,
        config: CityPipelineConfig,
        grid: SparseVoxelGrid,
        report: Callable,
    ) -> SparseVoxelGrid:
        """处理高度图 → 地形体素"""
        img = Image.open(config.heightmap_path).convert("L")
        hmap = np.array(img, dtype=np.float32) / 255.0

        h, w = hmap.shape
        scale = config.scale
        max_h = config.max_height

        # 如果需要缩放
        if scale != 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img_resized = img.resize((new_w, new_h), Image.BILINEAR)
            hmap = np.array(img_resized, dtype=np.float32) / 255.0
            h, w = hmap.shape

        water_y = int(config.water_level * max_h)

        logger.info("Heightmap: %d × %d, max_height=%d, water_level=%d", w, h, max_h, water_y)
        report(10.0, f"高度图: {w}×{h}")

        for z in range(h):
            if z % 50 == 0:
                pct = 10.0 + (z / h) * 35.0
                report(pct, f"生成地形 {z}/{h}…")

            for x in range(w):
                elevation = int(hmap[z, x] * max_h)

                # 地表层
                if elevation >= water_y:
                    grid.set(x, elevation, z, config.terrain_block)
                    # 地下层
                    for y in range(max(0, elevation - 3), elevation):
                        grid.set(x, y, z, config.underground_block)
                else:
                    # 水下
                    grid.set(x, elevation, z, config.underground_block)
                    for y in range(elevation + 1, water_y + 1):
                        grid.set(x, y, z, config.water_block)

        return grid

    def _process_buildings(
        self,
        config: CityPipelineConfig,
        grid: SparseVoxelGrid,
        report: Callable,
    ) -> SparseVoxelGrid:
        """处理建筑数据 — 简化版: 从 GeoJSON 读取 footprint + 高度"""
        path = Path(config.buildings_path)

        if path.suffix.lower() == ".geojson" or path.suffix.lower() == ".json":
            import json
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            features = data.get("features", [])
            logger.info("Processing %d building features", len(features))

            for fi, feature in enumerate(features):
                if fi % 100 == 0:
                    report(55.0 + (fi / max(1, len(features))) * 20.0,
                           f"建筑 {fi}/{len(features)}…")

                geom = feature.get("geometry", {})
                props = feature.get("properties", {})

                if geom.get("type") != "Polygon":
                    continue

                # 获取建筑高度 (尝试多个常见属性名)
                height = (
                    props.get("height")
                    or props.get("building:levels", 3) * 3
                    or 10
                )
                height = min(int(height), config.max_height)

                # 获取 polygon 坐标 (简化: 用 AABB)
                coords = geom.get("coordinates", [[]])[0]
                if not coords:
                    continue

                xs = [int(c[0] * config.scale) for c in coords]
                zs = [int(c[1] * config.scale) for c in coords]

                x_min, x_max = min(xs), max(xs)
                z_min, z_max = min(zs), max(zs)

                # 找地面高度
                base_y = 0
                for check_x in range(x_min, x_max + 1):
                    for check_z in range(z_min, z_max + 1):
                        existing = grid.get(check_x, 0, check_z)
                        if existing is not None:
                            # 找到该位置最高的方块
                            for cy in range(config.max_height, -1, -1):
                                if grid.get(check_x, cy, check_z) is not None:
                                    base_y = max(base_y, cy + 1)
                                    break

                # 放置建筑方块
                for bx in range(x_min, x_max + 1):
                    for bz in range(z_min, z_max + 1):
                        for by in range(base_y, base_y + height):
                            grid.set(bx, by, bz, config.building_block)

        else:
            logger.warning("Unsupported building format: %s", path.suffix)

        return grid

    @staticmethod
    def _get_xz_size(grid: SparseVoxelGrid) -> Tuple[int, int]:
        if grid.count == 0:
            return (0, 0)
        bmin, bmax = grid.bounds
        return (bmax[0] - bmin[0] + 1, bmax[2] - bmin[2] + 1)

    @staticmethod
    def _get_max_y(grid: SparseVoxelGrid) -> int:
        if grid.count == 0:
            return 0
        _, bmax = grid.bounds
        return bmax[1]
