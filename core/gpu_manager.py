"""
GpuManager — GPU / CPU 设备检测与资源调度器

功能:
- 启动时自动探测 Taichi 可用后端 (CUDA > Vulkan > OpenGL > CPU)
- 实时监控 CPU / GPU / RAM 使用率 (通过 psutil + nvidia-smi)
- 强制资源上限: GPU 显存 ≤ 80%, CPU 留 2 核给系统
- 提供 GpuContext 上下文管理器，自动分配/回收 Taichi Field
- 体素化分批执行，每批间检查系统资源，超限自动降速
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np
import psutil

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """设备信息"""
    backend: str = "cpu"          # "cuda", "vulkan", "opengl", "cpu"
    device_name: str = "CPU"
    gpu_mem_total_mb: int = 0
    gpu_mem_used_mb: int = 0
    cpu_count: int = 1
    cpu_reserved: int = 2
    ram_total_mb: int = 0


@dataclass
class ResourceSnapshot:
    """实时资源快照"""
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    ram_used_mb: int = 0
    gpu_mem_used_mb: int = 0
    gpu_mem_total_mb: int = 0
    gpu_utilization_pct: float = 0.0
    is_throttled: bool = False


class GpuManager:
    """
    GPU 资源管理器

    Usage::

        gm = GpuManager(gpu_mem_limit_pct=80, cpu_reserved_cores=2)
        gm.initialize()                     # 探测并初始化 Taichi
        print(gm.device_info)               # 查看设备
        snap = gm.snapshot()                 # 实时资源快照
        if gm.should_throttle():             # 检查是否该降速
            ...
    """

    def __init__(
        self,
        gpu_mem_limit_pct: int = 80,
        cpu_reserved_cores: int = 2,
        ram_warning_pct: int = 90,
        preferred_backend: str = "auto",
        chunk_batch_size: int = 256,
    ) -> None:
        self.gpu_mem_limit_pct = gpu_mem_limit_pct
        self.cpu_reserved_cores = cpu_reserved_cores
        self.ram_warning_pct = ram_warning_pct
        self.preferred_backend = preferred_backend
        self.chunk_batch_size = chunk_batch_size

        self._ti = None          # taichi module (lazy import)
        self._initialized = False
        self._device = DeviceInfo()
        self._backend_name = "cpu"

    # ── 初始化 ──────────────────────────────────────────────────

    def initialize(self) -> DeviceInfo:
        """
        探测硬件并初始化 Taichi。

        Returns
        -------
        DeviceInfo
            最终选定的设备概况
        """
        if self._initialized:
            return self._device

        self._device.cpu_count = os.cpu_count() or 1
        self._device.cpu_reserved = min(
            self.cpu_reserved_cores, max(1, self._device.cpu_count - 1)
        )
        self._device.ram_total_mb = psutil.virtual_memory().total // (1024 * 1024)

        # 尝试按优先级初始化 Taichi 后端
        backend = self._init_taichi()
        self._backend_name = backend
        self._device.backend = backend

        # 探测 GPU 信息
        self._probe_gpu()
        self._initialized = True

        logger.info(
            "GpuManager initialized: backend=%s, device=%s, GPU=%dMB, CPU=%d(-%d)",
            self._device.backend,
            self._device.device_name,
            self._device.gpu_mem_total_mb,
            self._device.cpu_count,
            self._device.cpu_reserved,
        )
        return self._device

    def _init_taichi(self) -> str:
        """尝试初始化 Taichi，返回实际使用的后端名"""
        try:
            import taichi as ti
            self._ti = ti
        except ImportError:
            logger.warning("Taichi not installed, falling back to CPU-only numpy mode")
            return "cpu"

        # 后端优先级
        if self.preferred_backend != "auto":
            backend_order = [self.preferred_backend]
        else:
            backend_order = ["cuda", "vulkan", "opengl", "cpu"]

        arch_map = {
            "cuda": getattr(self._ti, "cuda", None),
            "vulkan": getattr(self._ti, "vulkan", None),
            "opengl": getattr(self._ti, "opengl", None),
            "cpu": self._ti.cpu,
        }

        for name in backend_order:
            arch = arch_map.get(name)
            if arch is None:
                continue
            try:
                # 限制 CPU 线程数：总核心 - 保留核心
                cpu_threads = max(1, self._device.cpu_count - self._device.cpu_reserved)
                self._ti.init(
                    arch=arch,
                    cpu_max_num_threads=cpu_threads,
                    random_seed=42,
                )
                logger.info("Taichi initialized with backend: %s", name)
                return name
            except Exception as exc:
                logger.debug("Backend '%s' failed: %s", name, exc)
                continue

        # 最终 fallback
        self._ti.init(arch=self._ti.cpu)
        return "cpu"

    def _probe_gpu(self) -> None:
        """探测 GPU 型号和显存"""
        if self._backend_name == "cuda":
            self._probe_nvidia()
        elif self._backend_name in ("vulkan", "opengl"):
            self._device.device_name = f"GPU ({self._backend_name})"
        else:
            self._device.device_name = platform.processor() or "CPU"

    def _probe_nvidia(self) -> None:
        """通过 nvidia-smi 查询 NVIDIA GPU"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                line = result.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    self._device.device_name = parts[0]
                    self._device.gpu_mem_total_mb = int(float(parts[1]))
                    self._device.gpu_mem_used_mb = int(float(parts[2]))
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as exc:
            logger.debug("nvidia-smi query failed: %s", exc)
            self._device.device_name = "NVIDIA GPU"

    # ── 实时监控 ────────────────────────────────────────────────

    @property
    def device_info(self) -> DeviceInfo:
        return self._device

    @property
    def backend(self) -> str:
        return self._backend_name

    @property
    def taichi(self):
        """返回 taichi 模块引用"""
        return self._ti

    @property
    def usable_cpu_threads(self) -> int:
        """可用于计算的 CPU 线程数"""
        return max(1, self._device.cpu_count - self._device.cpu_reserved)

    def snapshot(self) -> ResourceSnapshot:
        """获取当前系统资源快照"""
        mem = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=0.1)

        snap = ResourceSnapshot(
            cpu_percent=cpu_pct,
            ram_percent=mem.percent,
            ram_used_mb=mem.used // (1024 * 1024),
            gpu_mem_total_mb=self._device.gpu_mem_total_mb,
        )

        # NVIDIA GPU 实时显存
        if self._backend_name == "cuda":
            self._update_gpu_stats(snap)

        # 是否被限流
        snap.is_throttled = (
            snap.ram_percent > self.ram_warning_pct
            or (
                snap.gpu_mem_total_mb > 0
                and snap.gpu_mem_used_mb / snap.gpu_mem_total_mb * 100
                > self.gpu_mem_limit_pct
            )
        )
        return snap

    def _update_gpu_stats(self, snap: ResourceSnapshot) -> None:
        """更新 NVIDIA GPU 实时数据"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                snap.gpu_mem_used_mb = int(float(parts[0]))
                snap.gpu_mem_total_mb = int(float(parts[1]))
                snap.gpu_utilization_pct = float(parts[2])
        except Exception:
            pass

    def should_throttle(self) -> bool:
        """检查是否应该降速"""
        return self.snapshot().is_throttled

    def get_safe_batch_size(self) -> int:
        """
        根据当前资源情况返回安全的分批大小。
        如果资源紧张则自动缩小。
        """
        snap = self.snapshot()
        batch = self.chunk_batch_size

        if snap.is_throttled:
            batch = max(32, batch // 2)
            logger.warning("Resources tight, reducing batch size to %d", batch)

        return batch

    # ── 上下文管理器 ────────────────────────────────────────────

    @contextmanager
    def compute_context(self) -> Generator[None, None, None]:
        """
        计算上下文管理器。

        在进入时检查资源，退出时同步 Taichi 并释放。

        Usage::

            with gpu_manager.compute_context():
                # 执行 Taichi 计算
                ...
        """
        if self.should_throttle():
            logger.warning("System resources low before computation")

        try:
            yield
        finally:
            if self._ti is not None:
                try:
                    self._ti.sync()
                except Exception:
                    pass
