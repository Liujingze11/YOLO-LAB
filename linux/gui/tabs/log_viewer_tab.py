"""
日志 & 结果查看 Tab — 浏览训练日志 CSV、实验结果目录。
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QHBoxLayout,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from paths import LOG_DIR, RESULTS_DIR, PREDICT_DIR
from gui.widgets import (
    btn,
    card,
    field_label,
    log_area,
    path_combo,
    path_combo_get,
    scroll_area,
    section_label,
    simple_combo,
    tiny_btn,
)
from gui.i18n import tr
from gui.utils import log_append, open_file_with_default_app, open_dir_safe, load_csv_log

ROOT = Path(__file__).resolve().parent.parent


class LogViewerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._path_history: dict[str, list[str]] = {}

        w = QWidget()
        w.setMinimumSize(560, 520)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(24, 16, 24, 24)
        outer.setSpacing(10)

        # ── 日志目录 ──
        card1, lay1 = card()
        lay1.addWidget(section_label("日志目录", i18n_key="logs.card.logdir"))
        lay1.addSpacing(14)
        self._path_history.setdefault("lv_logs", [])
        self.lv_log_dir = path_combo(default=LOG_DIR, history=self._path_history["lv_logs"])
        row1 = QHBoxLayout()
        row1.setSpacing(10)
        row1.addWidget(self.lv_log_dir, 1)
        b1 = btn("浏览", primary=False, i18n_key="train.btn.browse")
        b1.setFixedWidth(60)
        b1.clicked.connect(lambda: self._browse(self.lv_log_dir, True, None, "lv_logs"))
        row1.addWidget(b1)
        lay1.addLayout(row1)
        outer.addWidget(card1)

        # ── 日志文件选择 ──
        card2, lay2 = card()
        lay2.addWidget(section_label("历史日志文件", i18n_key="logs.card.files"))
        lay2.addSpacing(14)
        row2 = QHBoxLayout()
        row2.setSpacing(10)
        self.lv_csv_combo = simple_combo(min_width=280, font_size=13)
        self.lv_csv_combo.addItem(tr("logs.combo.csv_placeholder"))
        self.lv_csv_combo.activated.connect(self._on_lv_csv_selected)
        row2.addWidget(self.lv_csv_combo, 1)
        refresh_csv_btn = tiny_btn("⟳")
        refresh_csv_btn.clicked.connect(self._refresh_lv_csv_list)
        row2.addWidget(refresh_csv_btn)
        lay2.addLayout(row2)
        outer.addWidget(card2)

        # ── 日志内容 ──
        self.lv_log = log_area()
        outer.addWidget(self.lv_log, 1)

        # ── 实验结果 ──
        card3, lay3 = card()
        lay3.addWidget(section_label("实验 & 结果", i18n_key="logs.card.experiments"))
        lay3.addSpacing(14)

        exp_sel_row = QHBoxLayout()
        exp_sel_row.setSpacing(10)
        exp_sel_row.addWidget(field_label("实验", i18n_key="logs.field.experiment"))
        self.lv_exp_combo = simple_combo(min_width=200, font_size=13)
        self.lv_exp_combo.addItem(tr("logs.combo.exp_placeholder"))
        self.lv_exp_combo.activated.connect(self._on_lv_exp_selected)
        exp_sel_row.addWidget(self.lv_exp_combo, 1)
        exp_refresh = tiny_btn("⟳")
        exp_refresh.clicked.connect(self._refresh_lv_exp_list)
        exp_sel_row.addWidget(exp_refresh)
        lay3.addLayout(exp_sel_row)
        lay3.addSpacing(10)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self.lv_btn_exp_dir = btn("打开实验目录", primary=False, i18n_key="logs.btn.exp_dir")
        self.lv_btn_exp_dir.clicked.connect(self._open_lv_exp_dir)
        btn_row.addWidget(self.lv_btn_exp_dir)
        self.lv_btn_weights = btn("打开权重目录", primary=False, i18n_key="logs.btn.weights")
        self.lv_btn_weights.clicked.connect(self._open_lv_weights)
        btn_row.addWidget(self.lv_btn_weights)
        self.lv_btn_plot = btn("查看训练图表", primary=False, i18n_key="logs.btn.plot")
        self.lv_btn_plot.clicked.connect(self._open_lv_plot)
        btn_row.addWidget(self.lv_btn_plot)
        btn_row.addStretch()
        lay3.addLayout(btn_row)
        outer.addWidget(card3)

        # ── 快捷目录 ──
        card4, lay4 = card()
        lay4.addWidget(section_label("快捷目录", i18n_key="logs.card.quick"))
        lay4.addSpacing(14)
        quick_row = QHBoxLayout()
        quick_row.setSpacing(10)
        btn_specs = [
            ("训练结果", RESULTS_DIR, "logs.btn.results"),
            ("推理结果", str(Path(PREDICT_DIR) / "predict_result"), "logs.btn.predictions"),
            ("数据集", str(ROOT / "data" / "dataset"), "logs.btn.dataset"),
        ]
        for label, path, i18n_key in btn_specs:
            b = btn(label, primary=False, i18n_key=i18n_key)
            b.clicked.connect(lambda checked, p=path: open_dir_safe(p))
            quick_row.addWidget(b)
        quick_row.addStretch()
        lay4.addLayout(quick_row)
        outer.addWidget(card4)

        scroll = scroll_area(w)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def showEvent(self, e):
        super().showEvent(e)
        self._refresh_lv_csv_list()
        self._refresh_lv_exp_list()

    def _add_to_history(self, key, value):
        if not value:
            return
        hist = self._path_history.setdefault(key, [])
        if value in hist:
            hist.remove(value)
        hist.insert(0, value)
        if len(hist) > 20:
            hist.pop()

    def _refresh_combo_history(self, combo, history):
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(history)
        combo.blockSignals(False)

    def _browse(self, combo, directory, filter_str, hist_key):
        from PySide6.QtWidgets import QFileDialog
        start = Path(path_combo_get(combo) or str(ROOT)).resolve()
        if not start.is_dir() and not start.is_file():
            start = ROOT
        if directory:
            d = QFileDialog.getExistingDirectory(self, "选择目录", str(start))
            if d:
                combo.setCurrentText(d)
                self._add_to_history(hist_key, d)
                self._refresh_combo_history(combo, self._path_history[hist_key])
        else:
            f, _ = QFileDialog.getOpenFileName(self, "选择文件", str(start), filter_str or "所有文件 (*)")
            if f:
                combo.setCurrentText(f)
                self._add_to_history(hist_key, f)
                self._refresh_combo_history(combo, self._path_history[hist_key])

    def _refresh_lv_csv_list(self):
        combo = self.lv_csv_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(tr("logs.combo.csv_placeholder"))
        log_dir = path_combo_get(self.lv_log_dir)
        if Path(log_dir).is_dir():
            for f in sorted(Path(log_dir).glob("*.csv"), reverse=True):
                combo.addItem(f.name)
        combo.blockSignals(False)

    def _on_lv_csv_selected(self, idx: int):
        if idx <= 0:
            return
        text = self.lv_csv_combo.currentText()
        log_dir = path_combo_get(self.lv_log_dir)
        csv_path = Path(log_dir) / text
        if not csv_path.is_file():
            QMessageBox.warning(self, tr("msg.title.hint"), f"{tr('msg.yaml_not_found')}\n{csv_path}")
            return
        try:
            load_csv_log(self.lv_log, csv_path)
        except Exception as e:
            log_append(self.lv_log,
                f'<span style="color:#ff5555;">{tr("log.err_prefix")}</span>  {tr("msg.csv_read_failed")} {e}')

    def _refresh_lv_exp_list(self):
        combo = self.lv_exp_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(tr("logs.combo.exp_placeholder"))
        if Path(RESULTS_DIR).is_dir():
            for d in sorted(Path(RESULTS_DIR).iterdir(), reverse=True):
                if d.is_dir():
                    combo.addItem(d.name)
        combo.blockSignals(False)

    def _on_lv_exp_selected(self, idx: int):
        pass

    def _lv_exp_path(self):
        name = self.lv_exp_combo.currentText()
        placeholder = tr("logs.combo.exp_placeholder")
        if not name or name == placeholder:
            QMessageBox.warning(self, tr("msg.title.hint"), tr("msg.select_experiment"))
            return None
        return Path(RESULTS_DIR) / name

    def _open_lv_exp_dir(self):
        p = self._lv_exp_path()
        if p and p.is_dir():
            open_file_with_default_app(str(p))

    def _open_lv_weights(self):
        p = self._lv_exp_path()
        if p:
            wp = p / "weights"
            if wp.is_dir():
                open_file_with_default_app(str(wp))
            else:
                QMessageBox.warning(self, tr("msg.title.hint"), f"{tr('msg.weights_dir_not_found')}\n{wp}")

    def _open_lv_plot(self):
        p = self._lv_exp_path()
        if p:
            rp = p / "results.png"
            if rp.is_file():
                open_file_with_default_app(str(rp))
            else:
                QMessageBox.warning(self, tr("msg.title.hint"), tr("msg.plot_not_found"))
                if p.is_dir():
                    open_file_with_default_app(str(p))
