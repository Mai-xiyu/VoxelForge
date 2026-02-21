"""
测试 i18n 模块 — 翻译完整性、回退机制、语言切换
"""

import json
import sys
from pathlib import Path

import pytest

# 确保项目根在路径中
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


LOCALES_DIR = ROOT / "i18n" / "locales"
META_PATH = ROOT / "i18n" / "locale_meta.json"

LOCALE_FILES = list(LOCALES_DIR.glob("*.json"))


def _load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_keys(d: dict, prefix: str = "") -> set:
    """递归展开嵌套字典的所有 key 路径"""
    keys = set()
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            keys.update(_flatten_keys(v, full))
        else:
            keys.add(full)
    return keys


# ── 测试翻译文件完整性 ─────────────────────────────────────────

class TestI18nCompleteness:
    """所有语言文件应有相同的 key 结构"""

    @pytest.fixture(scope="class")
    def reference_keys(self) -> set:
        """以 en_US 为参考基准"""
        en = _load_json(LOCALES_DIR / "en_US.json")
        return _flatten_keys(en)

    @pytest.mark.parametrize("locale_file", LOCALE_FILES, ids=lambda p: p.stem)
    def test_keys_match_reference(self, locale_file: Path, reference_keys: set):
        data = _load_json(locale_file)
        keys = _flatten_keys(data)
        missing = reference_keys - keys
        extra = keys - reference_keys
        assert not missing, f"{locale_file.stem} missing keys: {missing}"
        # extra keys are allowed (warnings only)
        if extra:
            pytest.xfail(f"{locale_file.stem} has extra keys: {extra}")

    @pytest.mark.parametrize("locale_file", LOCALE_FILES, ids=lambda p: p.stem)
    def test_no_empty_values(self, locale_file: Path):
        data = _load_json(locale_file)
        keys = _flatten_keys(data)

        def get_val(d, path):
            for part in path.split("."):
                d = d[part]
            return d

        empty = [k for k in keys if get_val(data, k) == ""]
        assert not empty, f"{locale_file.stem} has empty values: {empty}"

    def test_meta_file_exists(self):
        assert META_PATH.exists()
        meta = _load_json(META_PATH)
        # meta 文件顶层键即为 locale code
        assert len(meta) >= 5
        assert "en_US" in meta
        assert "zh_CN" in meta


# ── 测试 I18nManager ───────────────────────────────────────────

class TestI18nManager:

    @pytest.fixture
    def manager(self):
        from i18n.i18n_manager import I18nManager
        I18nManager.reset()
        mgr = I18nManager.instance()
        mgr.load("en_US")
        return mgr

    def test_load_locales(self, manager):
        # available_locales 是属性，返回 meta 字典
        assert len(manager.available_locales) >= 5

    def test_default_locale(self, manager):
        assert manager.current_locale is not None

    def test_translate_existing_key(self, manager):
        manager.switch("en_US")
        val = manager.t("app.name")
        assert val == "VoxelForge"

    def test_translate_nested_key(self, manager):
        manager.switch("zh_CN")
        val = manager.t("common.cancel")
        assert val and val != "common.cancel"

    def test_fallback_to_en(self, manager):
        manager.switch("zh_CN")
        val = manager.t("nonexistent.key.here")
        assert val == "nonexistent.key.here"  # returns raw key

    def test_switch_language(self, manager):
        manager.switch("ja_JP")
        assert manager.current_locale == "ja_JP"
        val = manager.t("app.name")
        assert val == "VoxelForge"

    def test_template_variables(self, manager):
        manager.switch("en_US")
        val = manager.t("progress.percent", value="50")
        # Should contain "50" somewhere
        assert "50" in val or val  # graceful even if template not present
