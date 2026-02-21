"""
测试核心计算模块 — BlockMapper, SparseVoxelGrid, Voxelizer
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── SparseVoxelGrid ────────────────────────────────────────────

class TestSparseVoxelGrid:

    def test_set_get(self):
        from core.sparse_voxels import SparseVoxelGrid
        g = SparseVoxelGrid()
        g.set(0, 0, 0, "minecraft:stone")
        assert g.get(0, 0, 0) == "minecraft:stone"
        assert g.count == 1

    def test_remove(self):
        from core.sparse_voxels import SparseVoxelGrid
        g = SparseVoxelGrid()
        g.set(1, 2, 3, "minecraft:dirt")
        g.remove(1, 2, 3)
        assert g.get(1, 2, 3) is None
        assert g.count == 0

    def test_bounds(self):
        from core.sparse_voxels import SparseVoxelGrid
        g = SparseVoxelGrid()
        g.set(5, 10, 15, "a")
        g.set(20, 30, 40, "b")
        bmin, bmax = g.bounds
        assert bmin.tolist() == [5, 10, 15]
        assert bmax.tolist() == [20, 30, 40]

    def test_batch_set(self):
        from core.sparse_voxels import SparseVoxelGrid
        g = SparseVoxelGrid()
        coords = np.array([[x, 0, 0] for x in range(100)], dtype=np.int32)
        block_ids = [f"block_{x}" for x in range(100)]
        g.set_batch(coords, block_ids)
        assert g.count == 100
        assert g.get(50, 0, 0) == "block_50"

    def test_from_dense_grid(self):
        from core.sparse_voxels import SparseVoxelGrid
        # from_dense_grid expects int grid + palette list; 0 = air
        dense = np.zeros((4, 4, 4), dtype=np.int32)
        dense[0, 0, 0] = 1  # stone
        dense[3, 3, 3] = 2  # dirt
        palette = ["minecraft:air", "minecraft:stone", "minecraft:dirt"]
        g = SparseVoxelGrid.from_dense_grid(dense, palette)
        assert g.count == 2
        assert g.get(0, 0, 0) == "minecraft:stone"

    def test_block_statistics(self):
        from core.sparse_voxels import SparseVoxelGrid
        g = SparseVoxelGrid()
        g.set(0, 0, 0, "minecraft:stone")
        g.set(1, 0, 0, "minecraft:stone")
        g.set(2, 0, 0, "minecraft:dirt")
        stats = g.block_statistics()
        assert stats["minecraft:stone"] == 2
        assert stats["minecraft:dirt"] == 1

    def test_y_slice(self):
        from core.sparse_voxels import SparseVoxelGrid
        g = SparseVoxelGrid()
        g.set(5, 10, 5, "a")
        g.set(6, 10, 6, "b")
        g.set(7, 11, 7, "c")
        s = g.get_y_slice(10)
        assert len(s) == 2

    def test_iter_chunks(self):
        from core.sparse_voxels import SparseVoxelGrid
        g = SparseVoxelGrid()
        # 两个不同区块的体素
        g.set(0, 0, 0, "a")      # chunk (0, 0)
        g.set(16, 0, 0, "b")     # chunk (1, 0)
        chunks = list(g.iter_chunks())
        assert len(chunks) == 2


# ── BlockMapper ─────────────────────────────────────────────────

class TestBlockMapper:

    @pytest.fixture
    def mapper(self):
        from core.block_mapper import BlockMapper
        m = BlockMapper()
        m.load_palette_from_list([
            {"id": "minecraft:white_concrete", "rgb": [207, 213, 214], "category": "concrete"},
            {"id": "minecraft:red_concrete", "rgb": [142, 33, 33], "category": "concrete"},
            {"id": "minecraft:blue_concrete", "rgb": [44, 47, 143], "category": "concrete"},
            {"id": "minecraft:green_concrete", "rgb": [73, 91, 36], "category": "concrete"},
            {"id": "minecraft:black_concrete", "rgb": [8, 10, 15], "category": "concrete"},
            {"id": "minecraft:oak_planks", "rgb": [162, 130, 78], "category": "wood"},
        ])
        return m

    def test_palette_size(self, mapper):
        assert mapper.palette_size == 6

    def test_nearest_white(self, mapper):
        """纯白色应映射为 white_concrete"""
        from core.block_mapper import MapperConfig
        color_grid = np.zeros((1, 1, 1, 3), dtype=np.uint8)
        color_grid[0, 0, 0] = [255, 255, 255]
        occ = np.ones((1, 1, 1), dtype=np.int32)
        result = mapper.map_colors(color_grid, occ, MapperConfig(dithering=False))
        assert result[0, 0, 0] == "minecraft:white_concrete"

    def test_nearest_red(self, mapper):
        from core.block_mapper import MapperConfig
        color_grid = np.zeros((1, 1, 1, 3), dtype=np.uint8)
        color_grid[0, 0, 0] = [200, 30, 30]
        occ = np.ones((1, 1, 1), dtype=np.int32)
        result = mapper.map_colors(color_grid, occ, MapperConfig(dithering=False))
        assert result[0, 0, 0] == "minecraft:red_concrete"

    def test_category_filter(self, mapper):
        from core.block_mapper import MapperConfig
        # 只保留 wood 类别
        config = MapperConfig(
            dithering=False,
            category_whitelist={"wood"},
        )
        color_grid = np.zeros((1, 1, 1, 3), dtype=np.uint8)
        color_grid[0, 0, 0] = [255, 255, 255]
        occ = np.ones((1, 1, 1), dtype=np.int32)
        result = mapper.map_colors(color_grid, occ, config)
        assert result[0, 0, 0] == "minecraft:oak_planks"

    def test_dithering_runs(self, mapper):
        from core.block_mapper import MapperConfig
        color_grid = np.random.randint(0, 255, (8, 8, 8, 3), dtype=np.uint8)
        occ = np.ones((8, 8, 8), dtype=np.int32)
        result = mapper.map_colors(color_grid, occ, MapperConfig(dithering=True))
        # 应该返回非空结果
        non_empty = sum(1 for x in result.flat if x and x != "")
        assert non_empty > 0

    def test_rgb_to_lab_roundtrip(self):
        from core.block_mapper import BlockMapper
        for r, g, b in [(0, 0, 0), (255, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128)]:
            lab = BlockMapper._rgb_to_lab(r, g, b)
            rgb2 = BlockMapper._lab_to_rgb(*lab)
            assert abs(rgb2[0] - r) <= 2
            assert abs(rgb2[1] - g) <= 2
            assert abs(rgb2[2] - b) <= 2


# ── NBT 编码器 ──────────────────────────────────────────────────

class TestNBTEncoder:

    def test_encode_compound(self):
        from io_formats.nbt_encoder import NBTEncoder, nbt_int, nbt_string, nbt_short
        data = {
            "Version": nbt_int(2),
            "Name": nbt_string("Test"),
            "Width": nbt_short(16),
        }
        raw = NBTEncoder.encode_compound("TestRoot", data)
        assert isinstance(raw, bytes)
        assert len(raw) > 0
        # 第一字节应该是 TAG_COMPOUND (10)
        assert raw[0] == 10

    def test_varint_encode(self):
        from io_formats.nbt_encoder import PaletteEncoder
        assert PaletteEncoder.encode_varint(0) == b"\x00"
        assert PaletteEncoder.encode_varint(1) == b"\x01"
        assert PaletteEncoder.encode_varint(127) == b"\x7f"
        assert PaletteEncoder.encode_varint(128) == b"\x80\x01"
        assert PaletteEncoder.encode_varint(300) == b"\xac\x02"

    def test_palette_encoder(self):
        from io_formats.nbt_encoder import PaletteEncoder
        blocks = ["minecraft:stone", "minecraft:dirt", "minecraft:air", "minecraft:stone"]
        pmap, plist = PaletteEncoder.encode_palette(blocks)
        assert plist[0] == "minecraft:air"
        assert "minecraft:stone" in pmap
        assert len(plist) == 3  # unique blocks

    def test_pack_long_array(self):
        from io_formats.nbt_encoder import PaletteEncoder
        data = np.zeros(4096, dtype=np.int32)
        data[0] = 1
        data[1] = 2
        longs = PaletteEncoder.pack_long_array(data, 4)
        assert isinstance(longs, list)
        assert len(longs) > 0


# ── Voxelizer (CPU fallback) ────────────────────────────────────

class TestVoxelizer:

    def test_cube_voxelization(self):
        """测试简单立方体的体素化"""
        from core.gpu_manager import GpuManager
        from core.voxelizer import Voxelizer, VoxelMode

        # 创建一个简单的 2×2×2 立方体网格
        vertices = np.array([
            [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
            [0, 0, 10], [10, 0, 10], [10, 10, 10], [0, 10, 10],
        ], dtype=np.float32)

        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ], dtype=np.int32)

        gm = GpuManager()
        gm.initialize()
        vox = Voxelizer(gm)

        result = vox.voxelize(
            vertices=vertices,
            faces=faces,
            target_height=8,
            mode=VoxelMode.SURFACE,
        )

        assert result.voxel_count > 0
        assert result.grid.shape[1] == 8  # height should match target
