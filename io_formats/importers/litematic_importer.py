"""
Litematic 导入器 — 从 .litematic 文件导入 SparseVoxelGrid

使用 litemapy 库实现。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from core.sparse_voxels import SparseVoxelGrid

logger = logging.getLogger(__name__)


class LitematicImporter:
    """
    从 .litematic 文件导入为 SparseVoxelGrid。

    Usage::

        importer = LitematicImporter()
        grid = importer.load("my_build.litematic")
    """

    def __init__(self) -> None:
        self._litemapy = None
        try:
            import litemapy
            self._litemapy = litemapy
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._litemapy is not None

    def load(
        self,
        path: str | Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> SparseVoxelGrid:
        """加载 .litematic → SparseVoxelGrid"""
        if not self.available:
            raise RuntimeError("litemapy not installed")

        path = Path(path)
        lm = self._litemapy

        def report(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        report(0.0, "加载 Litematic…")

        schematic = lm.Schematic.load(str(path))
        grid = SparseVoxelGrid()

        total_blocks = 0
        for name, region in schematic.regions.items():
            report(20.0, f"读取区域: {name}")
            rx, ry, rz = region.xrange(), region.yrange(), region.zrange()

            for x in rx:
                for y in ry:
                    for z in rz:
                        block = region.getblock(x, y, z)
                        if block is not None and block.blockid != "minecraft:air":
                            # 重建完整 block state 字符串
                            bid = block.blockid
                            if block.properties:
                                props_str = ",".join(f"{k}={v}" for k, v in block.properties.items())
                                bid = f"{bid}[{props_str}]"
                            grid.set(
                                region.x + x,
                                region.y + y,
                                region.z + z,
                                bid,
                            )
                            total_blocks += 1

        logger.info("Imported Litematic: %s (%d blocks from %d regions)",
                     path.name, total_blocks, len(schematic.regions))
        report(100.0, f"导入完成: {total_blocks} 方块")

        return grid
