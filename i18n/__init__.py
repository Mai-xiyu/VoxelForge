"""
VoxelForge i18n — 国际化管理模块

提供全局翻译单例 I18nManager，支持：
- JSON 翻译文件加载（dot-path 嵌套键）
- 模板变量插值 t("progress.eta", minutes=3)
- 运行时动态切换语言（发射 Qt Signal）
- fallback 链: 当前语言 → en_US → 原始 key
- QWebChannel 联动（向嵌入的 HTML 面板推送语言数据）
"""

from i18n.i18n_manager import I18nManager

__all__ = ["I18nManager"]
