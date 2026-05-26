"""
YOLO 分割训练 / 推理桌面界面 — Apple 风格简约设计
启动：在项目根目录执行  python gui/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import tr, set_language, apply_language, current_lang, AVAILABLE_LANGS
from gui.styles import (
    COMBO_STYLE,
    DARK_TOGGLE_STYLE,
    FONT_FAMILIES,
    FONT_SIZE,
    TAB_WIDGET_STYLE,
    apply_theme_to_widgets,
)
from gui.tabs.train_tab import TrainTab
from gui.tabs.infer_tab import InferTab
from gui.tabs.tools_tab import ToolsTab
from gui.tabs.log_viewer_tab import LogViewerTab


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("app.title"))
        self._closing = False
        self.resize(820, 700)
        self.setMinimumSize(720, 520)

        self._dark_mode = False

        # ── Tab 组件 ──
        self._train_tab = TrainTab()
        self._infer_tab = InferTab()
        self._tools_tab = ToolsTab()
        self._log_viewer_tab = LogViewerTab()

        # ── Tab 关闭信号 ──
        self._train_tab.closing.connect(self._on_worker_closing)
        self._infer_tab.closing.connect(self._on_worker_closing)

        # ── Tab 容器 ──
        self._tabs = QTabWidget()
        self._tabs.setProperty("themeClass", "tab_widget")
        self._tabs.setStyleSheet(TAB_WIDGET_STYLE)
        tab_keys = ["tab.train", "tab.infer", "tab.logs", "tab.tools"]
        self._tabs.addTab(self._train_tab, tr(tab_keys[0]))
        self._tabs.tabBar().setTabData(0, tab_keys[0])
        self._tabs.addTab(self._infer_tab, tr(tab_keys[1]))
        self._tabs.tabBar().setTabData(1, tab_keys[1])
        self._tabs.addTab(self._log_viewer_tab, tr(tab_keys[2]))
        self._tabs.tabBar().setTabData(2, tab_keys[2])
        self._tabs.addTab(self._tools_tab, tr(tab_keys[3]))
        self._tabs.tabBar().setTabData(3, tab_keys[3])

        # ── Corner widget: 暗色模式 + 语言 ──
        self._dark_btn = QPushButton("☀")
        self._dark_btn.setProperty("themeClass", "tiny_btn")
        self._dark_btn.setStyleSheet(DARK_TOGGLE_STYLE)
        self._dark_btn.setFixedSize(32, 32)
        self._dark_btn.clicked.connect(self._toggle_dark_mode)

        self._lang_combo = QComboBox()
        self._lang_combo.setProperty("themeClass", "combo")
        self._lang_combo.setStyleSheet(COMBO_STYLE)
        self._lang_combo.setMaximumWidth(90)
        for code, name in AVAILABLE_LANGS.items():
            self._lang_combo.addItem(name, code)
        self._lang_combo.setCurrentIndex(0)
        self._lang_combo.currentIndexChanged.connect(self._on_lang_changed)

        corner = QWidget()
        cl = QHBoxLayout(corner)
        cl.setContentsMargins(0, 0, 8, 0)
        cl.setSpacing(6)
        cl.addWidget(self._lang_combo)
        cl.addWidget(self._dark_btn)
        self._tabs.setCornerWidget(corner)

        # ── 主布局 ──
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._tabs)

        # ── 快捷键 ──
        QShortcut(QKeySequence("Ctrl+Return"), self, self._on_ctrl_enter)

    # ── 暗色模式 ──

    def _toggle_dark_mode(self):
        self._dark_mode = not self._dark_mode
        self._dark_btn.setText("🌙" if self._dark_mode else "☀")
        apply_theme_to_widgets(self, self._dark_mode)

    # ── 语言切换 ──

    def _on_lang_changed(self, idx):
        lang = self._lang_combo.itemData(idx)
        if lang:
            set_language(lang)
            apply_language(self)

    # ── 快捷键 ──

    def _on_ctrl_enter(self):
        idx = self._tabs.currentIndex()
        if idx == 0:
            self._train_tab.on_ctrl_enter()
        elif idx == 1:
            self._infer_tab._on_start_infer()

    # ── 关闭处理 ──

    def _on_worker_closing(self):
        if self._closing:
            QApplication.quit()

    def closeEvent(self, event):
        if self._closing:
            event.accept()
            return
        self._closing = True
        self.hide()

        workers_running = False
        if self._train_tab.is_worker_running():
            self._train_tab.stop_worker()
            workers_running = True
        if self._infer_tab.is_worker_running():
            self._infer_tab.stop_worker()
            workers_running = True
        if self._tools_tab.is_worker_running():
            self._tools_tab.stop_worker()
            workers_running = True

        if not workers_running:
            QApplication.quit()
        event.ignore()


def main():
    app = QApplication(sys.argv)
    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    font.setPixelSize(FONT_SIZE)
    app.setFont(font)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


def _run_engine_mode():
    """Entry point when launched as --engine-{train,infer,tool} by a subprocess."""
    # Inject GPU bundle path so the subprocess can find CUDA torch
    from gui.gpu_manager import get_gpu_dir, has_gpu_ready_marker
    if has_gpu_ready_marker():
        gpu_dir = str(get_gpu_dir())
        if gpu_dir not in sys.path:
            sys.path.insert(0, gpu_dir)

    from gui import train_engine, infer_engine

    mode = sys.argv[1]  # e.g. "--engine-train"
    sys.argv.pop(1)

    if mode == "--engine-train":
        args = train_engine.parse_args()
        train_engine.run_non_interactive(args)
    elif mode == "--engine-infer":
        import argparse
        import gui.infer_engine as infer_engine
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default=infer_engine.BEST_SEG_MODEL)
        parser.add_argument("--source", default=infer_engine.TEST_IMAGES_DIR)
        parser.add_argument("--save-dir", default=str(Path(infer_engine.PREDICT_DIR) / "predict_result"))
        parser.add_argument("--conf", type=float, default=0.406)
        parser.add_argument("--imgsz", type=int, default=640)
        parser.add_argument("--lang", default="zh")
        args = parser.parse_args(sys.argv[1:])
        infer_engine._loc = infer_engine._load_locale(args.lang)
        cfg = infer_engine.InferConfig(
            model_path=args.model, source=args.source, save_dir=args.save_dir,
            conf=args.conf, imgsz=args.imgsz,
            task_param_file=str(infer_engine._ENGINE_DIR / "infer_task_params.json"),
            out_suffix="_overlay.jpg",
        )
        inferencer = infer_engine.YOLOInferencer(cfg)
        inferencer.run()
    elif mode == "--engine-tool":
        import json
        import importlib
        tool_args = json.loads(sys.argv[1])
        tool_idx = tool_args.pop("_tool_idx")

        TOOL_MODULES = [
            "tools.dataset_tools.create_empty_labels",
            "tools.dataset_tools.split_train_val.split_random_with_labels",
            "tools.dataset_tools.split_train_val_test.split_random_with_labels",
            "tools.dataset_tools.split_train_val.split_every_5th_with_labels",
            "tools.dataset_tools.split_images_only.split_random_images_only",
            "tools.dataset_tools.split_images_only.split_every_5th_images_only",
        ]
        mod = importlib.import_module(TOOL_MODULES[tool_idx])
        mod.run(**tool_args)
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("--engine-"):
        _run_engine_mode()
    else:
        main()
