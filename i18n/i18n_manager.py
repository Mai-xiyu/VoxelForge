"""
I18nManager — VoxelForge 国际化核心管理器

单例模式，全局唯一实例。
- 继承 QObject 以支持 Signal/Slot
- 加载 JSON locale 文件，支持 dot-path 嵌套键查找
- 运行时切换语言，通过 language_changed Signal 通知所有 UI 刷新
- fallback 链: current_locale → en_US → raw_key
"""

from __future__ import annotations

import json
import locale
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

# 本模块文件所在目录
_MODULE_DIR = Path(__file__).resolve().parent
_LOCALES_DIR = _MODULE_DIR / "locales"
_META_FILE = _MODULE_DIR / "locale_meta.json"


class I18nManager(QObject):
    """
    全局国际化管理器 (单例)

    Usage::

        from i18n import I18nManager

        i18n = I18nManager.instance()
        i18n.load("zh_CN")

        label = i18n.t("menu.file.open")                 # -> "打开文件"
        msg   = i18n.t("progress.eta", minutes=3)         # -> "预计剩余 3 分钟"
        i18n.switch("en_US")                               # 运行时切换
    """

    # ── Qt Signals ──────────────────────────────────────────────
    language_changed = Signal(str)  # 发射当前 locale code，如 "en_US"

    # ── 单例 ────────────────────────────────────────────────────
    _instance: Optional["I18nManager"] = None

    @classmethod
    def instance(cls) -> "I18nManager":
        """获取或创建全局唯一实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """重置单例（仅用于测试）"""
        cls._instance = None

    # ── 初始化 ──────────────────────────────────────────────────
    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)

        self._current_locale: str = "en_US"
        self._data: Dict[str, Any] = {}       # 当前语言的完整翻译字典
        self._fallback: Dict[str, Any] = {}   # en_US 的翻译字典（fallback）
        self._meta: Dict[str, Any] = {}       # locale_meta.json 内容

        # 预加载语言元信息
        self._load_meta()

        # 预加载英文 fallback
        self._fallback = self._read_locale_file("en_US")

    # ── 公开方法 ────────────────────────────────────────────────

    def load(self, locale_code: str) -> None:
        """
        加载指定语言。

        Parameters
        ----------
        locale_code : str
            语言代码，如 "zh_CN", "en_US", "ja_JP", "ko_KR", "ru_RU"。
            传入 "auto" 则自动检测系统语言。
        """
        if locale_code == "auto":
            locale_code = self._detect_system_locale()

        if locale_code == self._current_locale and self._data:
            return  # 已加载，跳过

        data = self._read_locale_file(locale_code)
        if not data:
            logger.warning(
                "Locale '%s' not found, falling back to en_US", locale_code
            )
            locale_code = "en_US"
            data = self._fallback

        self._data = data
        self._current_locale = locale_code
        logger.info("Language loaded: %s", locale_code)
        self.language_changed.emit(locale_code)

    def switch(self, locale_code: str) -> None:
        """运行时切换语言（别名 load）"""
        self.load(locale_code)

    def t(self, key: str, **kwargs: Any) -> str:
        """
        翻译函数。

        Parameters
        ----------
        key : str
            用 dot 分隔的嵌套键，如 "menu.file.open"
        **kwargs
            模板变量，如 minutes=3 会替换 "{minutes}"

        Returns
        -------
        str
            翻译后的字符串。未找到则 fallback 到 en_US，仍未找到返回原始 key。
        """
        # 在当前语言中查找
        result = self._resolve(self._data, key)

        # fallback 到 en_US
        if result is None:
            result = self._resolve(self._fallback, key)

        # 仍未找到 → 返回原始 key（开发期防崩溃）
        if result is None:
            logger.debug("Missing translation key: '%s' [%s]", key, self._current_locale)
            return key

        # 模板变量插值
        if kwargs:
            try:
                result = result.format(**kwargs)
            except (KeyError, IndexError, ValueError) as exc:
                logger.warning("Format error for key '%s': %s", key, exc)

        return result

    @property
    def current_locale(self) -> str:
        """当前语言代码"""
        return self._current_locale

    @property
    def available_locales(self) -> Dict[str, Dict[str, str]]:
        """
        返回所有可用语言的元信息。

        Returns
        -------
        dict
            {locale_code: {"name": ..., "native": ..., "flag": ..., "direction": ...}}
        """
        return dict(self._meta)

    def get_locale_data(self) -> Dict[str, Any]:
        """
        返回当前语言的完整翻译字典。
        主要供 QWebChannel 向 JS 端推送。
        """
        return dict(self._data)

    def reload(self) -> None:
        """热重载当前语言文件（开发期使用）"""
        self._data = self._read_locale_file(self._current_locale)
        self._fallback = self._read_locale_file("en_US")
        logger.info("Locale reloaded: %s", self._current_locale)
        self.language_changed.emit(self._current_locale)

    # ── 内部方法 ────────────────────────────────────────────────

    def _load_meta(self) -> None:
        """加载 locale_meta.json"""
        try:
            with open(_META_FILE, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.error("Failed to load locale_meta.json: %s", exc)
            self._meta = {}

    def _read_locale_file(self, locale_code: str) -> Dict[str, Any]:
        """读取并返回指定 locale 的 JSON 数据"""
        filepath = _LOCALES_DIR / f"{locale_code}.json"
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Locale file not found: %s", filepath)
            return {}
        except json.JSONDecodeError as exc:
            logger.error("JSON parse error in %s: %s", filepath, exc)
            return {}

    @staticmethod
    def _resolve(data: Dict[str, Any], dot_key: str) -> Optional[str]:
        """
        从嵌套字典中按 dot-path 查找值。

        Parameters
        ----------
        data : dict
            嵌套翻译字典
        dot_key : str
            如 "menu.file.open"

        Returns
        -------
        str or None
        """
        parts = dot_key.split(".")
        current: Any = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current if isinstance(current, str) else None

    @staticmethod
    def _detect_system_locale() -> str:
        """检测系统语言，返回对应的 locale code"""
        try:
            system_lang, _ = locale.getdefaultlocale()
        except ValueError:
            system_lang = None

        if not system_lang:
            return "en_US"

        # 映射常见的系统 locale → 我们的 locale code
        mapping = {
            "zh": "zh_CN",
            "zh_CN": "zh_CN",
            "zh_TW": "zh_CN",  # 简体 fallback
            "en": "en_US",
            "en_US": "en_US",
            "en_GB": "en_US",
            "ja": "ja_JP",
            "ja_JP": "ja_JP",
            "ko": "ko_KR",
            "ko_KR": "ko_KR",
            "ru": "ru_RU",
            "ru_RU": "ru_RU",
        }

        # 尝试完全匹配
        if system_lang in mapping:
            return mapping[system_lang]

        # 尝试语言前缀匹配
        lang_prefix = system_lang.split("_")[0]
        if lang_prefix in mapping:
            return mapping[lang_prefix]

        return "en_US"
