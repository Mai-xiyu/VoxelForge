"""
BlockMapper — 方块调色板颜色映射引擎

功能：
1. 加载 block_palette.json，将 RGB 转为 CIE L*a*b*
2. 构建 KD-Tree 用于最近邻颜色查找
3. Floyd-Steinberg 抖动 (XZ 层逐层扩散)
4. 类别白名单/黑名单过滤
5. 将体素颜色网格映射为 Minecraft block id 网格
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

# ── CIE L*a*b* 转换常量 ──
_D65_XN, _D65_YN, _D65_ZN = 0.950456, 1.0, 1.088754
_LAB_DELTA = 6.0 / 29.0
_LAB_DELTA2 = _LAB_DELTA ** 2
_LAB_DELTA3 = _LAB_DELTA ** 3


@dataclass
class BlockEntry:
    """调色板中一个方块的条目"""
    block_id: str
    rgb: Tuple[int, int, int]
    lab: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    category: str = "misc"


@dataclass
class MapperConfig:
    """映射器配置"""
    dithering: bool = True
    dither_strength: float = 1.0
    color_space: str = "lab"            # "lab" | "rgb"
    category_whitelist: Optional[Set[str]] = None
    category_blacklist: Optional[Set[str]] = None


class BlockMapper:
    """
    将 RGB 颜色网格映射为 Minecraft 方块 ID。

    Usage::

        mapper = BlockMapper()
        mapper.load_palette("config/block_palette.json")
        block_grid = mapper.map_colors(color_grid, config=MapperConfig(dithering=True))
    """

    def __init__(self) -> None:
        self._entries: List[BlockEntry] = []
        self._filtered: List[BlockEntry] = []
        self._tree: Optional[KDTree] = None
        self._lab_array: Optional[np.ndarray] = None
        self._rgb_array: Optional[np.ndarray] = None

    @property
    def palette_size(self) -> int:
        return len(self._entries)

    @property
    def active_size(self) -> int:
        return len(self._filtered)

    # ── 调色板加载 ──────────────────────────────────────────────

    def load_palette(self, path: str | Path) -> None:
        """从 JSON 文件加载方块调色板"""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._entries.clear()

        if "blocks" in data:
            # 扁平格式: blocks 数组，每个方块自带 category 字段
            for block in data["blocks"]:
                rgb = tuple(block["rgb"])
                entry = BlockEntry(
                    block_id=block["id"],
                    rgb=rgb,
                    lab=self._rgb_to_lab(rgb[0], rgb[1], rgb[2]),
                    category=block.get("category", "misc"),
                )
                self._entries.append(entry)
        else:
            # 嵌套格式 (向后兼容): categories 数组
            categories = data.get("categories", [])
            for cat in categories:
                cat_name = cat.get("name", "misc")
                for block in cat.get("blocks", []):
                    rgb = tuple(block["rgb"])
                    entry = BlockEntry(
                        block_id=block["id"],
                        rgb=rgb,
                        lab=self._rgb_to_lab(rgb[0], rgb[1], rgb[2]),
                        category=cat_name,
                    )
                    self._entries.append(entry)

        logger.info("Loaded %d blocks from palette: %s", len(self._entries), path.name)
        # 默认全部激活
        self._filtered = list(self._entries)
        self._build_tree()

    def load_palette_from_list(self, blocks: List[Dict]) -> None:
        """从字典列表加载调色板 (方便测试)"""
        self._entries.clear()
        for b in blocks:
            rgb = tuple(b["rgb"])
            entry = BlockEntry(
                block_id=b["id"],
                rgb=rgb,
                lab=self._rgb_to_lab(rgb[0], rgb[1], rgb[2]),
                category=b.get("category", "misc"),
            )
            self._entries.append(entry)
        self._filtered = list(self._entries)
        self._build_tree()

    # ── 过滤 ────────────────────────────────────────────────────

    def apply_filter(self, config: MapperConfig) -> None:
        """根据白/黑名单过滤可用方块"""
        filtered = self._entries

        if config.category_whitelist:
            filtered = [e for e in filtered if e.category in config.category_whitelist]
        if config.category_blacklist:
            filtered = [e for e in filtered if e.category not in config.category_blacklist]

        self._filtered = filtered
        self._build_tree()
        logger.info("Filtered palette: %d / %d blocks active", len(self._filtered), len(self._entries))

    # ── 核心映射 ────────────────────────────────────────────────

    def map_colors(
        self,
        color_grid: np.ndarray,
        occupancy: Optional[np.ndarray] = None,
        config: Optional[MapperConfig] = None,
    ) -> np.ndarray:
        """
        将颜色网格映射为方块 ID 索引网格。

        Parameters
        ----------
        color_grid : ndarray (X, Y, Z, 3) uint8
        occupancy : ndarray (X, Y, Z) optional
            如果提供，只映射占据体素
        config : MapperConfig

        Returns
        -------
        block_ids : ndarray (X, Y, Z) object (each cell is a block_id string or "")
        """
        if config is None:
            config = MapperConfig()

        if config.category_whitelist or config.category_blacklist:
            self.apply_filter(config)

        if not self._filtered:
            raise ValueError("No blocks in filtered palette — check filter settings")

        rx, ry, rz = color_grid.shape[:3]
        block_grid = np.empty((rx, ry, rz), dtype=object)
        block_grid[:] = ""

        use_lab = config.color_space == "lab"

        if config.dithering:
            block_grid = self._map_with_dithering(color_grid, occupancy, use_lab, config.dither_strength)
        else:
            block_grid = self._map_direct(color_grid, occupancy, use_lab)

        return block_grid

    def get_block_id(self, index: int) -> str:
        """根据索引返回方块 ID"""
        if 0 <= index < len(self._filtered):
            return self._filtered[index].block_id
        return ""

    def get_block_rgb(self, block_id: str) -> Optional[Tuple[int, int, int]]:
        """根据方块 ID 返回 RGB (用于预览)"""
        for e in self._entries:
            if e.block_id == block_id:
                return e.rgb
        return None

    # ── 直接映射 (无抖动) ───────────────────────────────────────

    def _map_direct(
        self,
        color_grid: np.ndarray,
        occupancy: Optional[np.ndarray],
        use_lab: bool,
    ) -> np.ndarray:
        rx, ry, rz = color_grid.shape[:3]
        block_grid = np.empty((rx, ry, rz), dtype=object)
        block_grid[:] = ""

        if occupancy is not None:
            coords = np.argwhere(occupancy > 0)
        else:
            coords = np.argwhere(np.any(color_grid > 0, axis=-1))

        if coords.shape[0] == 0:
            return block_grid

        # 批量提取颜色
        colors = color_grid[coords[:, 0], coords[:, 1], coords[:, 2]]  # (N, 3) uint8

        if use_lab:
            query_lab = np.array([self._rgb_to_lab(r, g, b) for r, g, b in colors])
            _, indices = self._tree.query(query_lab)
        else:
            _, indices = self._tree.query(colors.astype(float))

        for i in range(coords.shape[0]):
            x, y, z = coords[i]
            block_grid[x, y, z] = self._filtered[indices[i]].block_id

        return block_grid

    # ── Floyd-Steinberg 抖动映射 ────────────────────────────────

    def _map_with_dithering(
        self,
        color_grid: np.ndarray,
        occupancy: Optional[np.ndarray],
        use_lab: bool,
        strength: float,
    ) -> np.ndarray:
        """
        逐 Y 切片做 Floyd-Steinberg 抖动误差扩散
        在 XZ 平面上扩散，模拟从上往下俯瞰的视觉效果
        """
        rx, ry, rz = color_grid.shape[:3]
        block_grid = np.empty((rx, ry, rz), dtype=object)
        block_grid[:] = ""

        # 使用 float 颜色以便误差扩散
        float_colors = color_grid.astype(np.float32)

        for y in range(ry):
            occ_slice = occupancy[:, y, :] if occupancy is not None else None

            for x in range(rx):
                for z in range(rz):
                    if occ_slice is not None and occ_slice[x, z] == 0:
                        continue

                    r, g, b = float_colors[x, y, z]
                    if r == 0 and g == 0 and b == 0 and (occ_slice is None or occ_slice[x, z] == 0):
                        continue

                    # 当前像素颜色 → 最近方块
                    ri, gi, bi = int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(b, 0, 255))

                    if use_lab:
                        lab = self._rgb_to_lab(ri, gi, bi)
                        _, idx = self._tree.query(lab)
                    else:
                        _, idx = self._tree.query([float(ri), float(gi), float(bi)])

                    chosen = self._filtered[idx]
                    block_grid[x, y, z] = chosen.block_id

                    # 误差
                    err_r = r - chosen.rgb[0]
                    err_g = g - chosen.rgb[1]
                    err_b = b - chosen.rgb[2]

                    err = np.array([err_r, err_g, err_b]) * strength

                    # Floyd-Steinberg 扩散 (XZ 平面)
                    #        * 7/16
                    # 3/16  5/16  1/16
                    if z + 1 < rz:
                        float_colors[x, y, z + 1] += err * (7.0 / 16.0)
                    if x + 1 < rx:
                        if z - 1 >= 0:
                            float_colors[x + 1, y, z - 1] += err * (3.0 / 16.0)
                        float_colors[x + 1, y, z] += err * (5.0 / 16.0)
                        if z + 1 < rz:
                            float_colors[x + 1, y, z + 1] += err * (1.0 / 16.0)

        return block_grid

    # ── KD-Tree 构建 ────────────────────────────────────────────

    def _build_tree(self) -> None:
        """构建用于最近邻查找的 KD-Tree"""
        if not self._filtered:
            self._tree = None
            return

        lab_arr = np.array([e.lab for e in self._filtered])
        rgb_arr = np.array([e.rgb for e in self._filtered], dtype=float)

        self._lab_array = lab_arr
        self._rgb_array = rgb_arr
        self._tree = KDTree(lab_arr)

        logger.debug("KD-Tree built with %d entries (Lab space)", len(self._filtered))

    # ── 颜色空间转换 ───────────────────────────────────────────

    @staticmethod
    def _rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
        """RGB → CIE L*a*b* (D65 光源)"""
        # sRGB → 线性 RGB
        def linearize(c: float) -> float:
            c /= 255.0
            if c > 0.04045:
                return ((c + 0.055) / 1.055) ** 2.4
            else:
                return c / 12.92

        rl = linearize(float(r))
        gl = linearize(float(g))
        bl = linearize(float(b))

        # RGB → XYZ (sRGB D65 矩阵)
        x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
        y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
        z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041

        # XYZ → Lab
        def f(t: float) -> float:
            if t > _LAB_DELTA3:
                return t ** (1.0 / 3.0)
            else:
                return t / (3.0 * _LAB_DELTA2) + 4.0 / 29.0

        fx = f(x / _D65_XN)
        fy = f(y / _D65_YN)
        fz = f(z / _D65_ZN)

        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b_val = 200.0 * (fy - fz)

        return (L, a, b_val)

    @staticmethod
    def _lab_to_rgb(L: float, a: float, b_val: float) -> Tuple[int, int, int]:
        """CIE L*a*b* → RGB (用于预览调试)"""
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b_val / 200.0

        def inv_f(t: float) -> float:
            if t > _LAB_DELTA:
                return t ** 3
            else:
                return 3.0 * _LAB_DELTA2 * (t - 4.0 / 29.0)

        x = _D65_XN * inv_f(fx)
        y = _D65_YN * inv_f(fy)
        z = _D65_ZN * inv_f(fz)

        # XYZ → 线性 RGB
        rl = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
        gl = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
        bl = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

        def delinearize(c: float) -> int:
            if c > 0.0031308:
                c = 1.055 * (c ** (1.0 / 2.4)) - 0.055
            else:
                c = 12.92 * c
            return int(np.clip(c * 255.0, 0, 255))

        return (delinearize(rl), delinearize(gl), delinearize(bl))

    # ── 调色板统计与调试 ────────────────────────────────────────

    def get_categories(self) -> Dict[str, int]:
        """返回所有类别及其方块数量"""
        cats: Dict[str, int] = {}
        for e in self._entries:
            cats[e.category] = cats.get(e.category, 0) + 1
        return cats

    def preview_palette(self, max_entries: int = 20) -> List[Dict]:
        """返回当前活跃调色板的前 N 个条目（调试用）"""
        result = []
        for e in self._filtered[:max_entries]:
            result.append({
                "id": e.block_id,
                "rgb": list(e.rgb),
                "lab": [round(c, 2) for c in e.lab],
                "category": e.category,
            })
        return result
