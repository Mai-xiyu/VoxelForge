"""
Schematic 导出器 — Sponge Schematic v2 格式 (.schem / .schematic)

Sponge Schematic v2 结构:
    Schematic (Compound)
    ├── Version: 2 (Int)
    ├── DataVersion: MC 数据版本 (Int)
    ├── Width: X (Short)
    ├── Height: Y (Short)
    ├── Length: Z (Short)
    ├── Palette (Compound)
    │   ├── "minecraft:stone": 0 (Int)
    │   └── ...
    ├── PaletteMax: N (Int)
    ├── BlockData: VarInt[] (Byte Array)
    └── Metadata (Compound)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from core.sparse_voxels import SparseVoxelGrid
from io_formats.nbt_encoder import (
    NBTEncoder, NBTTag, PaletteEncoder,
    TAG_BYTE_ARRAY, TAG_COMPOUND, TAG_INT, TAG_SHORT, TAG_STRING,
    nbt_byte_array, nbt_compound, nbt_int, nbt_short, nbt_string,
)

logger = logging.getLogger(__name__)

# MC 1.20.4 data version
DEFAULT_DATA_VERSION = 3700


class SchematicExporter:
    """
    Sponge Schematic v2 导出器

    Usage::

        exporter = SchematicExporter()
        exporter.export(
            grid=sparse_voxel_grid,
            output_path="build.schem",
        )
    """

    def __init__(self, data_version: int = DEFAULT_DATA_VERSION) -> None:
        self.data_version = data_version

    def export(
        self,
        grid: SparseVoxelGrid,
        output_path: str | Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        """
        将 SparseVoxelGrid 导出为 .schem 文件
        """
        output_path = Path(output_path)

        def report(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        report(0.0, "准备导出 Schematic…")

        # 计算边界
        if grid.count == 0:
            raise ValueError("Empty grid, nothing to export")

        bmin, bmax = grid.bounds
        width = bmax[0] - bmin[0] + 1   # X
        height = bmax[1] - bmin[1] + 1   # Y
        length = bmax[2] - bmin[2] + 1   # Z

        logger.info("Schematic dimensions: %d × %d × %d", width, height, length)

        if width > 32767 or height > 32767 or length > 32767:
            raise ValueError(f"Dimensions exceed Short limit: {width}×{height}×{length}")

        report(10.0, f"尺寸: {width}×{height}×{length}")

        # 收集所有 block id 并创建 palette
        all_block_ids = set()
        all_block_ids.add("minecraft:air")
        for _, block_id in grid._data.items():
            if isinstance(block_id, str):
                all_block_ids.add(block_id)
            elif isinstance(block_id, int):
                all_block_ids.add(f"minecraft:block_{block_id}")

        palette_map, palette_list = PaletteEncoder.encode_palette(list(all_block_ids))

        report(30.0, f"调色板: {len(palette_list)} 种方块")

        # 构建 block data (按 Y, Z, X 顺序遍历)
        indices = []
        total = width * height * length
        for y in range(height):
            for z in range(length):
                for x in range(width):
                    wx = x + bmin[0]
                    wy = y + bmin[1]
                    wz = z + bmin[2]

                    block_id = grid.get(wx, wy, wz)
                    if block_id is not None:
                        if isinstance(block_id, str):
                            idx = palette_map.get(block_id, 0)
                        else:
                            idx = palette_map.get(f"minecraft:block_{block_id}", 0)
                    else:
                        idx = 0  # air

                    indices.append(idx)

            if y % 16 == 0:
                pct = 30.0 + (y / height) * 50.0
                report(pct, f"编码层 {y}/{height}…")

        report(80.0, "编码 BlockData…")

        # VarInt 编码
        block_data = PaletteEncoder.pack_block_data_varint(indices)

        # 构建 NBT

        # Palette compound
        palette_tags = {}
        for bid, idx in palette_map.items():
            palette_tags[bid] = nbt_int(idx)

        # Metadata
        metadata_tags = {
            "Generator": nbt_string("VoxelForge"),
        }

        # Root compound
        root_tags = {
            "Version": nbt_int(2),
            "DataVersion": nbt_int(self.data_version),
            "Width": nbt_short(width),
            "Height": nbt_short(height),
            "Length": nbt_short(length),
            "PaletteMax": nbt_int(len(palette_list)),
            "Palette": nbt_compound(palette_tags),
            "BlockData": nbt_byte_array(block_data),
            "Metadata": nbt_compound(metadata_tags),
        }

        report(90.0, "写入文件…")

        raw = NBTEncoder.encode_compound("Schematic", root_tags)
        NBTEncoder.write_gzipped(raw, str(output_path))

        logger.info("Exported Schematic: %s (%d blocks, %d palette entries)",
                     output_path.name, grid.count, len(palette_list))
        report(100.0, f"导出完成: {output_path.name}")
