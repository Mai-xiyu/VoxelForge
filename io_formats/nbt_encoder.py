"""
NBT 编码器 — Minecraft NBT 二进制格式编/解码

支持所有 NBT 标签类型:
TAG_End(0), TAG_Byte(1), TAG_Short(2), TAG_Int(3), TAG_Long(4),
TAG_Float(5), TAG_Double(6), TAG_Byte_Array(7), TAG_String(8),
TAG_List(9), TAG_Compound(10), TAG_Int_Array(11), TAG_Long_Array(12)

也提供高层 API 将 SparseVoxelGrid 转为 NBT compound 结构。
"""

from __future__ import annotations

import gzip
import io
import logging
import struct
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── NBT 标签类型 ID ──────────────────────────────────────────
TAG_END = 0
TAG_BYTE = 1
TAG_SHORT = 2
TAG_INT = 3
TAG_LONG = 4
TAG_FLOAT = 5
TAG_DOUBLE = 6
TAG_BYTE_ARRAY = 7
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10
TAG_INT_ARRAY = 11
TAG_LONG_ARRAY = 12


class NBTEncoder:
    """
    低级 NBT 二进制编码器。

    Usage::

        nbt = NBTEncoder()
        data = {
            "schematicVersion": NBTTag(TAG_INT, 2),
            "Width": NBTTag(TAG_SHORT, 64),
            "BlockData": NBTTag(TAG_BYTE_ARRAY, bytes_data),
        }
        raw = nbt.encode_compound("Schematic", data)
        nbt.write_gzipped(raw, "output.nbt")
    """

    @staticmethod
    def encode_compound(name: str, tags: Dict[str, "NBTTag"]) -> bytes:
        """编码一个命名的 Compound 标签"""
        buf = io.BytesIO()
        # 写入 Compound 标签头
        buf.write(struct.pack(">b", TAG_COMPOUND))
        name_bytes = name.encode("utf-8")
        buf.write(struct.pack(">H", len(name_bytes)))
        buf.write(name_bytes)
        # 写入内容
        NBTEncoder._write_compound_payload(buf, tags)
        return buf.getvalue()

    @staticmethod
    def write_gzipped(data: bytes, path: str) -> None:
        """将编码的 NBT 数据 gzip 压缩后写入文件"""
        with gzip.open(path, "wb") as f:
            f.write(data)
        logger.info("Wrote gzipped NBT: %s (%d bytes)", path, len(data))

    @staticmethod
    def write_raw(data: bytes, path: str) -> None:
        """写入未压缩的 NBT 数据"""
        with open(path, "wb") as f:
            f.write(data)

    # ── 内部编码方法 ────────────────────────────────────────────

    @staticmethod
    def _write_compound_payload(buf: BinaryIO, tags: Dict[str, "NBTTag"]) -> None:
        for key, tag in tags.items():
            NBTEncoder._write_named_tag(buf, key, tag)
        buf.write(struct.pack(">b", TAG_END))

    @staticmethod
    def _write_named_tag(buf: BinaryIO, name: str, tag: "NBTTag") -> None:
        buf.write(struct.pack(">b", tag.tag_type))
        name_bytes = name.encode("utf-8")
        buf.write(struct.pack(">H", len(name_bytes)))
        buf.write(name_bytes)
        NBTEncoder._write_payload(buf, tag)

    @staticmethod
    def _write_payload(buf: BinaryIO, tag: "NBTTag") -> None:
        t = tag.tag_type
        v = tag.value

        if t == TAG_BYTE:
            buf.write(struct.pack(">b", v))
        elif t == TAG_SHORT:
            buf.write(struct.pack(">h", v))
        elif t == TAG_INT:
            buf.write(struct.pack(">i", v))
        elif t == TAG_LONG:
            buf.write(struct.pack(">q", v))
        elif t == TAG_FLOAT:
            buf.write(struct.pack(">f", v))
        elif t == TAG_DOUBLE:
            buf.write(struct.pack(">d", v))
        elif t == TAG_BYTE_ARRAY:
            data = v if isinstance(v, (bytes, bytearray)) else bytes(v)
            buf.write(struct.pack(">i", len(data)))
            buf.write(data)
        elif t == TAG_STRING:
            s = v.encode("utf-8")
            buf.write(struct.pack(">H", len(s)))
            buf.write(s)
        elif t == TAG_LIST:
            elem_type, elements = v
            buf.write(struct.pack(">b", elem_type))
            buf.write(struct.pack(">i", len(elements)))
            for elem in elements:
                NBTEncoder._write_payload(buf, NBTTag(elem_type, elem))
        elif t == TAG_COMPOUND:
            if isinstance(v, dict):
                NBTEncoder._write_compound_payload(buf, v)
            else:
                raise TypeError(f"TAG_COMPOUND value must be dict, got {type(v)}")
        elif t == TAG_INT_ARRAY:
            arr = v if isinstance(v, (list, tuple)) else list(v)
            buf.write(struct.pack(">i", len(arr)))
            for x in arr:
                buf.write(struct.pack(">i", x))
        elif t == TAG_LONG_ARRAY:
            arr = v if isinstance(v, (list, tuple)) else list(v)
            buf.write(struct.pack(">i", len(arr)))
            for x in arr:
                buf.write(struct.pack(">q", x))
        else:
            raise ValueError(f"Unknown tag type: {t}")


class NBTTag:
    """NBT 标签值包装"""
    __slots__ = ("tag_type", "value")

    def __init__(self, tag_type: int, value: Any) -> None:
        self.tag_type = tag_type
        self.value = value

    def __repr__(self) -> str:
        return f"NBTTag(type={self.tag_type}, value={self.value!r})"


# ── 便捷构造函数 ────────────────────────────────────────────────

def nbt_byte(v: int) -> NBTTag:
    return NBTTag(TAG_BYTE, v)

def nbt_short(v: int) -> NBTTag:
    return NBTTag(TAG_SHORT, v)

def nbt_int(v: int) -> NBTTag:
    return NBTTag(TAG_INT, v)

def nbt_long(v: int) -> NBTTag:
    return NBTTag(TAG_LONG, v)

def nbt_float(v: float) -> NBTTag:
    return NBTTag(TAG_FLOAT, v)

def nbt_double(v: float) -> NBTTag:
    return NBTTag(TAG_DOUBLE, v)

def nbt_string(v: str) -> NBTTag:
    return NBTTag(TAG_STRING, v)

def nbt_byte_array(v: bytes | bytearray | list) -> NBTTag:
    return NBTTag(TAG_BYTE_ARRAY, v)

def nbt_int_array(v: list) -> NBTTag:
    return NBTTag(TAG_INT_ARRAY, v)

def nbt_long_array(v: list) -> NBTTag:
    return NBTTag(TAG_LONG_ARRAY, v)

def nbt_list(elem_type: int, elements: list) -> NBTTag:
    return NBTTag(TAG_LIST, (elem_type, elements))

def nbt_compound(v: Dict[str, NBTTag]) -> NBTTag:
    return NBTTag(TAG_COMPOUND, v)


# ── 方块调色板编码工具 ─────────────────────────────────────────

class PaletteEncoder:
    """
    将方块 ID 字符串集合编码为 palette + 压缩 block data。
    用于 Schematic / Litematic 等格式。
    """

    @staticmethod
    def encode_palette(block_ids: List[str]) -> Tuple[Dict[str, int], List[str]]:
        """
        创建方块调色板映射。
        
        Returns
        -------
        (palette_map, palette_list)
            palette_map: block_id → index
            palette_list: 有序列表
        """
        unique = list(dict.fromkeys(block_ids))  # 保持顺序
        # 确保 minecraft:air 在索引 0
        if "minecraft:air" not in unique:
            unique.insert(0, "minecraft:air")
        elif unique[0] != "minecraft:air":
            unique.remove("minecraft:air")
            unique.insert(0, "minecraft:air")

        palette_map = {bid: idx for idx, bid in enumerate(unique)}
        return palette_map, unique

    @staticmethod
    def encode_varint(value: int) -> bytes:
        """编码一个 VarInt (用于 Sponge Schematic 格式)"""
        result = bytearray()
        while True:
            byte = value & 0x7F
            value >>= 7
            if value != 0:
                byte |= 0x80
            result.append(byte)
            if value == 0:
                break
        return bytes(result)

    @staticmethod
    def pack_block_data_varint(indices: List[int]) -> bytes:
        """将索引列表编码为 VarInt 字节数组"""
        result = bytearray()
        for idx in indices:
            result.extend(PaletteEncoder.encode_varint(idx))
        return bytes(result)

    @staticmethod
    def pack_long_array(data: np.ndarray, bits_per_entry: int) -> List[int]:
        """
        将数据打包为 MC 格式的 long array。
        用于 1.16+ 的 Section block states。
        """
        entries_per_long = 64 // bits_per_entry
        mask = (1 << bits_per_entry) - 1
        flat = data.flatten()
        n_longs = (len(flat) + entries_per_long - 1) // entries_per_long

        longs = []
        for i in range(n_longs):
            val = 0
            for j in range(entries_per_long):
                idx = i * entries_per_long + j
                if idx < len(flat):
                    val |= (int(flat[idx]) & mask) << (j * bits_per_entry)
            # 转为有符号 long
            if val >= (1 << 63):
                val -= (1 << 64)
            longs.append(val)

        return longs
