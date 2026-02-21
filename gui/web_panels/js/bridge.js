/**
 * VoxelForge Web Panel — QWebChannel 通信桥接
 *
 * 提供 Python ↔ JS 双向通信能力。
 */

(function () {
    "use strict";

    let i18nBridge = null;

    /**
     * 初始化 QWebChannel 连接
     */
    function initBridge() {
        if (typeof QWebChannel === "undefined") {
            console.warn("QWebChannel not available — running standalone");
            return;
        }

        new QWebChannel(qt.webChannelTransport, function (channel) {
            i18nBridge = channel.objects.i18n || null;
            if (i18nBridge) {
                console.log("i18n bridge connected");
                // 监听语言切换
                if (i18nBridge.language_changed) {
                    i18nBridge.language_changed.connect(function (locale) {
                        console.log("Language switched to:", locale);
                        onLanguageChanged(locale);
                    });
                }
            }
        });
    }

    /**
     * 语言切换回调 — 可在此更新 DOM 文本
     */
    function onLanguageChanged(locale) {
        // 未来可以在这里通过 i18nBridge 获取翻译并更新页面
        var evt = new CustomEvent("voxelforge:lang", { detail: { locale: locale } });
        document.dispatchEvent(evt);
    }

    /**
     * 供 Python 调用: 更新状态栏文本
     */
    window.updateStatus = function (text) {
        var el = document.getElementById("status-text");
        if (el) el.textContent = text;
    };

    /**
     * 供 Python 调用: 更新 GPU 内存限制显示
     */
    window.updateGpuLimit = function (pct) {
        var el = document.getElementById("gpu-limit");
        if (el) el.textContent = "≤ " + pct + "%";
    };

    /**
     * 供 Python 调用: 更新统计信息
     */
    window.updateStats = function (stats) {
        // stats = { blocks: number, elapsed: number, output: string }
        var evt = new CustomEvent("voxelforge:stats", { detail: stats });
        document.dispatchEvent(evt);
    };

    // DOM 就绪后初始化
    document.addEventListener("DOMContentLoaded", initBridge);
})();
