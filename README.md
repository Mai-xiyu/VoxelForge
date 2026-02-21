# VoxelForge

**多源 3D 数据 → Minecraft 区块文件** 的 GPU 加速转换平台。

支持将 3D 模型、城市 GIS 数据、激光扫描点云及 3D 高斯泼溅 (3DGS) 数据转换为
Minecraft `.mca`(Anvil)、`.schematic`（Sponge v2）和 `.litematic`（Litematica）格式。

---

## 功能亮点

| 特性 | 说明 |
|------|------|
| 🔥 **GPU 体素化** | 基于 Taichi，自动检测 CUDA / Vulkan / OpenGL / CPU 后端 |
| 🎨 **CIE L\*a\*b\* 色彩匹配** | KD-Tree 最近邻 + Floyd-Steinberg 抖动，~150 种方块调色板 |
| 🏙️ **三条转换管线** | 3D 模型(A)、城市高度图/GeoJSON(B)、扫描点云/3DGS(C) |
| 🧊 **稀疏体素存储** | 基于字典的稀疏网格，MC 区块/区域迭代器，内存高效 |
| 🖥️ **混合 GUI** | PySide6 原生控件 + PyVista 3D 视口 + QWebEngineView 美化面板 |
| 🌏 **i18n 多语言** | 简体中文 / English / 日本語 / 한국어 / Русский，运行时热切换 |
| 🛡️ **资源安全** | GPU 显存 ≤ 80%、CPU 保留 2 核、RAM 90% 预警，分批计算 |

---

## 项目结构

```
VoxelForge/
├── main.py                  # 应用入口
├── pyproject.toml           # 依赖与构建配置
├── config/
│   ├── settings.yaml        # 全局配置（GPU 限制、体素化参数等）
│   └── block_palette.json   # ~150 种 MC 方块 RGB 调色板
├── core/
│   ├── gpu_manager.py       # Taichi 多后端初始化 + 资源监控
│   ├── sparse_voxels.py     # 稀疏体素网格 & MC 区块迭代
│   ├── voxelizer.py         # GPU 加速表面/实心体素化
│   ├── block_mapper.py      # L*a*b* 颜色→方块映射 + 抖动
│   ├── mesh_processor.py    # 网格导入(OBJ/FBX/GLTF/PLY/STL 等)
│   └── point_cloud.py       # 点云/3DGS PLY 处理 + 泊松重建
├── io_formats/
│   ├── nbt_encoder.py       # 原生 NBT 二进制编码器（13 种标签）
│   ├── schematic_exporter.py # Sponge Schematic v2 导出
│   ├── litematic_exporter.py # Litematica 导入/导出
│   └── mca_exporter.py      # Anvil .mca 区域文件导出
├── pipelines/
│   ├── pipeline_model.py    # 路径 A：3D 模型→MC
│   ├── pipeline_city.py     # 路径 B：城市/GIS→MC
│   └── pipeline_scan.py     # 路径 C：扫描/点云→MC
├── gui/
│   ├── main_window.py       # PySide6 主窗口 & 管线配置
│   ├── viewport_3d.py       # PyVista/VTK 3D 视口
│   ├── widgets.py           # 进度条、调色板视图、日志面板等
│   └── web/
│       └── index.html       # Catppuccin Mocha 主题信息面板
├── i18n/
│   ├── i18n_manager.py      # I18nManager（加载/切换/翻译/回退）
│   ├── locale_meta.json     # 语言元信息
│   └── locales/             # 5 种语言 JSON
│       ├── zh_CN.json
│       ├── en_US.json
│       ├── ja_JP.json
│       ├── ko_KR.json
│       └── ru_RU.json
└── tests/
    ├── test_i18n.py         # i18n 完整性 & 管理器单元测试
    └── test_core.py         # 体素网格、方块映射、NBT 编码测试
```

---

## 环境要求

- **Python** ≥ 3.10
- **操作系统**：Windows / macOS / Linux
- （可选）NVIDIA GPU + CUDA 或 Vulkan 支持的显卡以获得最佳性能

---

## 安装

```bash
cd VoxelForge

# 安装运行时依赖 + 开发工具
pip install -e ".[dev]"
```

### 主要依赖

| 类别 | 库 |
|------|----|
| GPU 计算 | `taichi` |
| GUI | `PySide6`, `pyvistaqt`, `pyvista` |
| 网格处理 | `trimesh`, `open3d`, `Pillow` |
| MC 格式 | `litemapy`, `nbtlib`, `amulet-core` |
| 科学计算 | `numpy`, `scipy`, `scikit-learn` |
| 工具 | `psutil`, `pyyaml` |

---

## 快速开始

### 启动 GUI

```bash
python main.py
```

### 命令行管线示例（Python API）

```python
from core import GpuManager, Voxelizer, VoxelMode, BlockMapper, MeshProcessor
from io_formats.litematic_exporter import LitematicExporter

# 1. 初始化 GPU
gpu = GpuManager()
gpu.initialize()

# 2. 加载并处理网格
mp = MeshProcessor()
mesh = mp.load("model.obj")
mesh = mp.repair(mesh)

# 3. 体素化
vox = Voxelizer(gpu)
result = vox.voxelize(mesh.vertices, mesh.faces, target_height=128, mode=VoxelMode.SURFACE)

# 4. 颜色→方块映射
bm = BlockMapper()
bm.load_palette("config/block_palette.json")
grid = bm.map_colors(result.color_grid, result.grid, bm.default_config())

# 5. 导出 .litematic
from core.sparse_voxels import SparseVoxelGrid
sg = SparseVoxelGrid.from_dense_grid(grid)
LitematicExporter().export(sg, "output.litematic", name="MyModel")
```

---

## 三条转换管线

### A — 3D 模型

支持格式：`.obj` `.fbx` `.gltf` `.glb` `.ply` `.stl` `.dxf` `.3ds` `.dae` `.off` `.3mf`

加载 → 修复 → 简化 → 缩放 → 体素化 → 色彩映射 → 导出

### B — 城市 / GIS

- 高度图（灰度 PNG/TIFF）→ 地形层（地表/地下/水面）
- GeoJSON 建筑轮廓 → 高度拉伸

### C — 扫描 / 点云 / 3DGS

支持格式：`.ply` `.pcd` `.xyz` `.las` `.e57`

自动检测 3DGS PLY（球谐系数 SH DC → RGB），可选泊松曲面重建或直接体素化。

---

## 输出格式

| 格式 | 扩展名 | 用途 |
|------|--------|------|
| Anvil Region | `.mca` | 直接放入存档 `region/` 目录 |
| Sponge Schematic v2 | `.schematic` | WorldEdit 导入 |
| Litematica | `.litematic` | Litematica Mod 投影 |

目标版本：**Minecraft Java 1.20+**（data version 3700，高度范围 -64 ~ 319）

---

## 运行测试

```bash
python -m pytest tests/ -v
```

---

## 配置

编辑 `config/settings.yaml` 调整：

- `compute.gpu_memory_limit_pct`：GPU 显存使用上限（默认 80%）
- `compute.cpu_reserved_cores`：保留 CPU 核心数（默认 2）
- `voxelizer.default_height`：默认体素化高度
- `block_mapper.color_space`：颜色空间（`lab` / `rgb`）
- `block_mapper.dithering`：Floyd-Steinberg 抖动开关
- `minecraft.target_version`：目标 MC 版本
- `language`：界面语言（`auto` / `zh_CN` / `en_US` / `ja_JP` / `ko_KR` / `ru_RU`）

---

## 许可证

MIT License
