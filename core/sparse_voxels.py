"""
SparseVoxelGrid — 稀疏体素存储引擎

核心数据结构：
- 内部使用 Dict[Tuple[int,int,int], str] 存储非空体素 (坐标 → block_id)
- 空气不存储，城市级场景实际存储仅占总体积 1-5%
- 提供按 MC 架构切分: Region(512×512) → Chunk(16×16) → Section(16×16×16)
- 内置隐藏面检测：标记被6邻域完全包围的内部方块（可选跳过输出）
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 类型别名
Coord = Tuple[int, int, int]


@dataclass
class Section:
    """一个 MC Section (16×16×16)"""
    y_index: int
    blocks: Dict[Coord, str] = field(default_factory=dict)

    @property
    def palette(self) -> List[str]:
        """生成该 Section 的方块调色板（去重有序列表）"""
        unique = ["minecraft:air"]
        seen = {"minecraft:air"}
        for block_id in self.blocks.values():
            if block_id not in seen:
                seen.add(block_id)
                unique.append(block_id)
        return unique

    @property
    def is_empty(self) -> bool:
        return len(self.blocks) == 0


@dataclass
class Chunk:
    """一个 MC Chunk (16×N×16)，包含多个 Section"""
    cx: int  # chunk x (世界坐标 // 16)
    cz: int  # chunk z
    sections: Dict[int, Section] = field(default_factory=dict)

    def get_or_create_section(self, y_index: int) -> Section:
        if y_index not in self.sections:
            self.sections[y_index] = Section(y_index=y_index)
        return self.sections[y_index]


@dataclass
class Region:
    """一个 MC Region (32×32 chunks)，对应一个 .mca 文件"""
    rx: int  # region x (chunk_x // 32)
    rz: int  # region z
    chunks: Dict[Tuple[int, int], Chunk] = field(default_factory=dict)

    def get_or_create_chunk(self, cx: int, cz: int) -> Chunk:
        key = (cx, cz)
        if key not in self.chunks:
            self.chunks[key] = Chunk(cx=cx, cz=cz)
        return self.chunks[key]

    @property
    def filename(self) -> str:
        return f"r.{self.rx}.{self.rz}.mca"


class SparseVoxelGrid:
    """
    稀疏体素网格

    Usage::

        grid = SparseVoxelGrid()
        grid.set(10, 64, 20, "minecraft:stone")
        grid.set(10, 65, 20, "minecraft:grass_block")

        block = grid.get(10, 64, 20)           # -> "minecraft:stone"
        block = grid.get(0, 0, 0)              # -> None (空气)

        for region in grid.iter_regions():
            write_mca(region)
    """

    def __init__(self) -> None:
        self._voxels: Dict[Coord, str] = {}
        self._bounds_min: Optional[np.ndarray] = None
        self._bounds_max: Optional[np.ndarray] = None

    # ── 基本操作 ────────────────────────────────────────────────

    def set(self, x: int, y: int, z: int, block_id: str) -> None:
        """设置体素"""
        self._voxels[(x, y, z)] = block_id
        self._invalidate_bounds()

    def get(self, x: int, y: int, z: int) -> Optional[str]:
        """获取体素，空气返回 None"""
        return self._voxels.get((x, y, z))

    def remove(self, x: int, y: int, z: int) -> None:
        """移除体素"""
        self._voxels.pop((x, y, z), None)
        self._invalidate_bounds()

    @property
    def count(self) -> int:
        """非空体素数量"""
        return len(self._voxels)

    @property
    def is_empty(self) -> bool:
        return len(self._voxels) == 0

    def clear(self) -> None:
        self._voxels.clear()
        self._bounds_min = None
        self._bounds_max = None

    # ── 批量操作 ────────────────────────────────────────────────

    def set_batch(self, coords: np.ndarray, block_ids: List[str]) -> None:
        """
        批量设置体素。

        Parameters
        ----------
        coords : ndarray shape (N, 3) int
        block_ids : list[str] 长度 N
        """
        for i in range(len(block_ids)):
            x, y, z = int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])
            self._voxels[(x, y, z)] = block_ids[i]
        self._invalidate_bounds()

    def set_batch_single_block(self, coords: np.ndarray, block_id: str) -> None:
        """批量设置同一种方块"""
        for i in range(coords.shape[0]):
            x, y, z = int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])
            self._voxels[(x, y, z)] = block_id
        self._invalidate_bounds()

    # ── 从 numpy 体素网格导入 ──────────────────────────────────

    @classmethod
    def from_dense_grid(
        cls,
        grid: np.ndarray,
        block_ids: List[str],
        origin: Tuple[int, int, int] = (0, 0, 0),
    ) -> "SparseVoxelGrid":
        """
        从 dense numpy 网格创建稀疏体素。

        Parameters
        ----------
        grid : ndarray shape (X, Y, Z) dtype int
            每个元素是 block_ids 列表的索引；0 = 空气
        block_ids : list[str]
            调色板列表，index 0 = air
        origin : tuple
            世界坐标原点偏移

        Returns
        -------
        SparseVoxelGrid
        """
        obj = cls()
        ox, oy, oz = origin

        # 找到所有非零体素
        nonzero = np.argwhere(grid > 0)

        for idx in range(nonzero.shape[0]):
            xi, yi, zi = nonzero[idx]
            block_idx = int(grid[xi, yi, zi])
            if 0 < block_idx < len(block_ids):
                obj._voxels[(int(xi) + ox, int(yi) + oy, int(zi) + oz)] = block_ids[block_idx]

        return obj

    # ── 包围盒 ──────────────────────────────────────────────────

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (min_xyz, max_xyz)"""
        if self._bounds_min is None:
            self._compute_bounds()
        return self._bounds_min, self._bounds_max

    @property
    def size(self) -> Tuple[int, int, int]:
        """(width_x, height_y, depth_z)"""
        bmin, bmax = self.bounds
        d = bmax - bmin + 1
        return (int(d[0]), int(d[1]), int(d[2]))

    def _compute_bounds(self) -> None:
        if not self._voxels:
            self._bounds_min = np.zeros(3, dtype=int)
            self._bounds_max = np.zeros(3, dtype=int)
            return

        coords = np.array(list(self._voxels.keys()), dtype=np.int32)
        self._bounds_min = coords.min(axis=0)
        self._bounds_max = coords.max(axis=0)

    def _invalidate_bounds(self) -> None:
        self._bounds_min = None
        self._bounds_max = None

    # ── 隐藏面检测 ──────────────────────────────────────────────

    def find_hidden_voxels(self) -> Set[Coord]:
        """
        找到被6个不透明邻居完全包围的内部体素。

        Returns
        -------
        set of Coord
            可以跳过输出的内部体素坐标集合
        """
        offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        hidden = set()

        for (x, y, z) in self._voxels:
            all_neighbors_solid = True
            for dx, dy, dz in offsets:
                if (x + dx, y + dy, z + dz) not in self._voxels:
                    all_neighbors_solid = False
                    break
            if all_neighbors_solid:
                hidden.add((x, y, z))

        logger.info(
            "Hidden face analysis: %d / %d voxels are internal (%.1f%%)",
            len(hidden), len(self._voxels),
            len(hidden) / max(1, len(self._voxels)) * 100,
        )
        return hidden

    # ── MC 结构化切分 ───────────────────────────────────────────

    def iter_regions(self) -> Iterator[Region]:
        """
        按 Minecraft Region → Chunk → Section 架构切分体素。

        Yields
        ------
        Region
            每个 Region 对应一个 .mca 文件
        """
        # 先按 region 分组
        region_map: Dict[Tuple[int, int], Region] = {}

        for (x, y, z), block_id in self._voxels.items():
            cx = x >> 4           # x // 16
            cz = z >> 4           # z // 16
            rx = cx >> 5          # cx // 32
            rz = cz >> 5          # cz // 32
            sy = y >> 4           # y // 16 (section y index)

            # 获取 region
            rkey = (rx, rz)
            if rkey not in region_map:
                region_map[rkey] = Region(rx=rx, rz=rz)
            region = region_map[rkey]

            # 获取 chunk
            chunk = region.get_or_create_chunk(cx, cz)

            # 获取 section
            section = chunk.get_or_create_section(sy)

            # 存储局部坐标
            local_x = x & 0xF  # x % 16
            local_y = y & 0xF  # y % 16
            local_z = z & 0xF  # z % 16
            section.blocks[(local_x, local_y, local_z)] = block_id

        yield from region_map.values()

    def iter_chunks(self) -> Iterator[Chunk]:
        """直接迭代所有 Chunk"""
        for region in self.iter_regions():
            yield from region.chunks.values()

    # ── 切片视图 ────────────────────────────────────────────────

    def get_y_slice(self, y: int) -> Dict[Tuple[int, int], str]:
        """获取指定 Y 层的水平切片"""
        result = {}
        for (vx, vy, vz), block_id in self._voxels.items():
            if vy == y:
                result[(vx, vz)] = block_id
        return result

    # ── 统计信息 ────────────────────────────────────────────────

    def block_statistics(self) -> Dict[str, int]:
        """统计各种方块的数量"""
        stats: Dict[str, int] = defaultdict(int)
        for block_id in self._voxels.values():
            stats[block_id] += 1
        return dict(sorted(stats.items(), key=lambda x: -x[1]))

    def __repr__(self) -> str:
        if self.is_empty:
            return "SparseVoxelGrid(empty)"
        sx, sy, sz = self.size
        return f"SparseVoxelGrid(blocks={self.count}, size={sx}×{sy}×{sz})"
