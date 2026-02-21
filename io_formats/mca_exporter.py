"""
MCA (Anvil) 导出器 — Minecraft 1.20+ 区域文件格式

.mca 文件结构:
- 每个 .mca 文件覆盖 32×32 个区块 (一个 Region)
- 每个区块 (Chunk) 16×384×16 方块 (Y: -64..319)
- 每个区块分 24 个 Section (每个 16×16×16)
- Section 内使用 palette + packed long array 压缩

文件头:
- 0x0000-0x0FFF: 位置表 (1024 × 4 bytes)
- 0x1000-0x1FFF: 时间戳表 (1024 × 4 bytes)
- 0x2000+: 压缩的区块 NBT 数据
"""

from __future__ import annotations

import io
import logging
import math
import struct
import time
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from core.sparse_voxels import SparseVoxelGrid
from io_formats.nbt_encoder import (
    NBTEncoder, NBTTag, PaletteEncoder,
    TAG_BYTE, TAG_BYTE_ARRAY, TAG_COMPOUND, TAG_INT, TAG_LIST,
    TAG_LONG, TAG_LONG_ARRAY, TAG_SHORT, TAG_STRING,
    nbt_byte, nbt_compound, nbt_int, nbt_list, nbt_long,
    nbt_long_array, nbt_short, nbt_string,
)

logger = logging.getLogger(__name__)

# MC 1.20.4
DEFAULT_DATA_VERSION = 3700
SECTION_SIZE = 16
CHUNK_HEIGHT = 384  # -64..319
MIN_Y = -64
MAX_Y = 319
SECTIONS_PER_CHUNK = CHUNK_HEIGHT // SECTION_SIZE  # 24


class McaExporter:
    """
    Minecraft Anvil (.mca) 区域文件导出器

    Usage::

        exporter = McaExporter()
        exporter.export(
            grid=sparse_voxel_grid,
            output_dir="world/region/",
            base_y=64,
        )
    """

    def __init__(self, data_version: int = DEFAULT_DATA_VERSION) -> None:
        self.data_version = data_version

    def export(
        self,
        grid: SparseVoxelGrid,
        output_dir: str | Path,
        base_y: int = 64,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[str]:
        """
        将 SparseVoxelGrid 导出为 .mca 文件。

        Parameters
        ----------
        grid : SparseVoxelGrid
        output_dir : 存放 .mca 文件的目录
        base_y : 模型底部在世界中的 Y 坐标
        progress_callback : 进度回调

        Returns
        -------
        List[str]: 生成的文件路径列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def report(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        if grid.count == 0:
            raise ValueError("Empty grid, nothing to export")

        report(0.0, "分析区域结构…")

        # 偏移体素坐标到世界坐标
        bmin, _ = grid.bounds
        offset_y = base_y - bmin[1]

        # 收集所有涉及的 region
        regions: Dict[Tuple[int, int], Dict[Tuple[int, int], Dict]] = {}

        for (x, y, z), block_id in grid._data.items():
            # 世界坐标
            wy = y + offset_y

            # 跳过超出高度限制的方块
            if wy < MIN_Y or wy > MAX_Y:
                continue

            chunk_x = x >> 4
            chunk_z = z >> 4
            region_x = chunk_x >> 5
            region_z = chunk_z >> 5

            rkey = (region_x, region_z)
            ckey = (chunk_x, chunk_z)

            if rkey not in regions:
                regions[rkey] = {}
            if ckey not in regions[rkey]:
                regions[rkey][ckey] = {}

            # 区块内 local 坐标
            local_x = x & 0xF
            local_z = z & 0xF
            regions[rkey][ckey][(local_x, wy, local_z)] = block_id

        n_regions = len(regions)
        logger.info("Exporting to %d .mca files", n_regions)
        report(10.0, f"需要导出 {n_regions} 个区域文件")

        # 逐 region 生成 .mca
        generated = []
        for ri, ((rx, rz), chunks) in enumerate(regions.items()):
            filename = f"r.{rx}.{rz}.mca"
            filepath = output_dir / filename

            pct_base = 10.0 + (ri / n_regions) * 85.0
            report(pct_base, f"生成 {filename}…")

            self._write_mca(filepath, chunks, rx, rz)
            generated.append(str(filepath))

        report(100.0, f"导出完成: {len(generated)} 个 .mca 文件")
        return generated

    def _write_mca(
        self,
        path: Path,
        chunks: Dict[Tuple[int, int], Dict],
        region_x: int,
        region_z: int,
    ) -> None:
        """写入一个 .mca 区域文件"""

        # 准备区块数据
        chunk_data_list: Dict[Tuple[int, int], bytes] = {}

        for (cx, cz), blocks in chunks.items():
            nbt_bytes = self._build_chunk_nbt(cx, cz, blocks)
            # zlib 压缩
            compressed = zlib.compress(nbt_bytes)
            chunk_data_list[(cx, cz)] = compressed

        # 构建 .mca 文件
        locations = bytearray(4096)   # 1024 entries × 4 bytes
        timestamps = bytearray(4096)  # 1024 entries × 4 bytes

        chunk_payloads = bytearray()
        current_sector = 2  # 前两个 sector 是 header

        ts = int(time.time())

        for (cx, cz), data in chunk_data_list.items():
            # Region 内索引
            local_cx = cx & 0x1F
            local_cz = cz & 0x1F
            idx = (local_cx + local_cz * 32) * 4

            # 构建 payload: length(4) + compression_type(1) + data
            payload = struct.pack(">I", len(data) + 1) + struct.pack(">B", 2) + data

            # 对齐到 4096 字节 sector
            sectors_needed = math.ceil(len(payload) / 4096)
            padded = payload.ljust(sectors_needed * 4096, b"\x00")

            # 位置表: offset(3 bytes) + sector count(1 byte)
            offset_bytes = struct.pack(">I", current_sector)[1:]  # 取后3字节
            locations[idx:idx + 3] = offset_bytes
            locations[idx + 3] = sectors_needed

            # 时间戳
            timestamps[idx:idx + 4] = struct.pack(">I", ts)

            chunk_payloads.extend(padded)
            current_sector += sectors_needed

        # 写入文件
        with open(path, "wb") as f:
            f.write(locations)
            f.write(timestamps)
            f.write(chunk_payloads)

        logger.info("Wrote .mca: %s (%d chunks)", path.name, len(chunk_data_list))

    def _build_chunk_nbt(
        self,
        chunk_x: int,
        chunk_z: int,
        blocks: Dict[Tuple[int, int, int], str],
    ) -> bytes:
        """构建一个区块的 NBT 数据"""

        # 按 section 分组
        sections_data: Dict[int, Dict[Tuple[int, int, int], str]] = {}
        for (local_x, world_y, local_z), block_id in blocks.items():
            section_y = (world_y - MIN_Y) // SECTION_SIZE
            section_local_y = (world_y - MIN_Y) % SECTION_SIZE

            if section_y not in sections_data:
                sections_data[section_y] = {}
            sections_data[section_y][(local_x, section_local_y, local_z)] = block_id

        # 构建 section NBT list
        section_tags = []
        for sy in range(SECTIONS_PER_CHUNK):
            section_tag = self._build_section_nbt(sy, sections_data.get(sy, {}))
            section_tags.append(section_tag)

        # Chunk root
        root_tags = {
            "DataVersion": nbt_int(self.data_version),
            "xPos": nbt_int(chunk_x),
            "zPos": nbt_int(chunk_z),
            "yPos": nbt_int(MIN_Y // SECTION_SIZE),
            "Status": nbt_string("minecraft:full"),
            "sections": nbt_list(TAG_COMPOUND, section_tags),
        }

        return NBTEncoder.encode_compound("", root_tags)

    def _build_section_nbt(
        self,
        section_index: int,
        blocks: Dict[Tuple[int, int, int], str],
    ) -> Dict[str, NBTTag]:
        """构建一个 Section (16×16×16) 的 NBT"""
        section_y = section_index + (MIN_Y // SECTION_SIZE)

        if not blocks:
            # 空 section — 仅含 air
            palette_tags = [nbt_compound({"Name": nbt_string("minecraft:air")})]
            return {
                "Y": nbt_byte(section_y & 0xFF if section_y >= 0 else (256 + section_y) & 0xFF),
                "block_states": nbt_compound({
                    "palette": nbt_list(TAG_COMPOUND, palette_tags),
                }),
            }

        # 收集 palette
        unique_blocks = ["minecraft:air"]
        for bid in blocks.values():
            bid_str = str(bid)
            if bid_str not in unique_blocks:
                unique_blocks.append(bid_str)

        palette_map = {bid: idx for idx, bid in enumerate(unique_blocks)}
        bits = max(4, math.ceil(math.log2(len(unique_blocks)))) if len(unique_blocks) > 1 else 4

        # 填充 4096 个方块索引 (XZY 顺序)
        indices = np.zeros(4096, dtype=np.int32)
        for (lx, ly, lz), bid in blocks.items():
            idx = (ly * SECTION_SIZE + lz) * SECTION_SIZE + lx
            if 0 <= idx < 4096:
                indices[idx] = palette_map.get(str(bid), 0)

        # 打包为 long array
        longs = PaletteEncoder.pack_long_array(indices, bits)

        # 构建 palette tags
        palette_tags = []
        for bid in unique_blocks:
            base_id = bid
            props_dict = {}
            if "[" in bid and bid.endswith("]"):
                base_id = bid[:bid.index("[")]
                props_str = bid[bid.index("[") + 1:-1]
                for pair in props_str.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        props_dict[k.strip()] = v.strip()

            entry = {"Name": nbt_string(base_id)}
            if props_dict:
                entry["Properties"] = nbt_compound({
                    k: nbt_string(v) for k, v in props_dict.items()
                })
            palette_tags.append(entry)

        section_tag = {
            "Y": nbt_byte(section_y & 0xFF if section_y >= 0 else (256 + section_y) & 0xFF),
            "block_states": nbt_compound({
                "palette": nbt_list(TAG_COMPOUND, palette_tags),
                "data": nbt_long_array(longs),
            }),
        }

        return section_tag
