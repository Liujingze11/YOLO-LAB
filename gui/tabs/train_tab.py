"""
训练 Tab — YOLO 模型训练的完整配置界面。
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.config import TrainConfig
from gui.device import get_available_devices, get_default_device
from gui.train_engine import list_experiments
from gui.i18n import tr, current_lang
from gui.styles import CHECKBOX_STYLE, COMBO_STYLE, RADIO_STYLE
from gui.utils import (
    engine_cmd,
    log_append,
    model_file_ok,
    open_file_with_default_app,
    load_presets,
    save_presets,
)
from gui.widgets import (
    btn,
    card,
    danger_btn,
    field_label,
    input_,
    log_area,
    path_combo,
    path_combo_get,
    progress_bar,
    scroll_area,
    section_label,
    simple_combo,
    spinner,
    tiny_btn,
)
from gui.model_selector import ModelSelector
from gui.workers import TrainWorker


class TrainTab(QWidget):
    closing = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._train_worker: TrainWorker | None = None
        self._presets = load_presets()
        self._path_history: dict[str, list[str]] = {}

        w = QWidget()
        w.setMinimumSize(640, 920)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(24, 16, 24, 24)
        outer.setSpacing(10)

        # ── 路径卡片 ──
        card1, lay1 = card()
        header1 = QHBoxLayout()
        header1.addWidget(section_label("路径", i18n_key="train.card.paths"))
        header1.addStretch()
        scan_models_btn = tiny_btn("扫描模型", i18n_key="train.btn.scan")
        scan_models_btn.clicked.connect(self._scan_trained_models)
        header1.addWidget(scan_models_btn)
        edit_yaml_btn = tiny_btn("编辑 data.yaml", i18n_key="train.btn.edit_yaml")
        edit_yaml_btn.clicked.connect(self._open_data_yaml)
        header1.addWidget(edit_yaml_btn)
        lay1.addLayout(header1)
        lay1.addSpacing(14)

        for key in ["data_yaml", "model", "results", "logs"]:
            self._path_history.setdefault(key, [])

        self.tr_data_yaml = path_combo(default="", history=self._path_history["data_yaml"])
        self.tr_model = ModelSelector()
        self.tr_results   = path_combo(default="", history=self._path_history["results"])
        self.tr_logs      = path_combo(default="", history=self._path_history["logs"])

        rows_data = [
            ("data.yaml", self.tr_data_yaml, "data_yaml", False, "YAML (*.yaml *.yml)"),
            ("结果目录", self.tr_results,   "results",    True,  None),
            ("日志目录", self.tr_logs,      "logs",       True,  None),
        ]
        for label, cb, hist_key, is_dir, flt in rows_data:
            row = QHBoxLayout()
            row.setSpacing(10)
            lbl = field_label(label, i18n_key={
                "data.yaml": "train.data_yaml",
                "结果目录": "train.field.results_dir",
                "日志目录": "train.field.log_dir",
            }.get(label, ""))
            lbl.setFixedWidth(72)
            row.addWidget(lbl)
            row.addWidget(cb, 1)
            b = btn("浏览", primary=False, i18n_key="train.btn.browse")
            b.setFixedWidth(60)
            b.clicked.connect(lambda checked, c=cb, d=is_dir, f=flt, k=hist_key: self._browse(c, d, f, k))
            row.addWidget(b)
            lay1.addLayout(row)
            lay1.addSpacing(8)

        # Dataset info hint
        self.dataset_info = QLabel("")
        self.dataset_info.setStyleSheet("font-size: 11px; color: #8e8e93; padding-left: 80px;")
        self.dataset_info.setWordWrap(True)
        lay1.addWidget(self.dataset_info)
        self.tr_data_yaml.lineEdit().editingFinished.connect(self._update_dataset_info)
        self.tr_data_yaml.currentIndexChanged.connect(self._update_dataset_info)

        model_row = QHBoxLayout()
        model_row.setSpacing(10)
        model_lbl = field_label("初始权重", i18n_key="train.field.init_weights")
        model_lbl.setFixedWidth(72)
        model_row.addWidget(model_lbl)
        model_row.addWidget(self.tr_model, 1)
        lay1.addLayout(model_row)
        lay1.addSpacing(8)
        outer.addWidget(card1)

        # ── 超参数卡片 ──
        card2, lay2 = card()
        lay2.addWidget(section_label("超参数", i18n_key="train.card.hyperparams"))
        lay2.addSpacing(14)

        self.tr_epochs = spinner(1, 100000, 150, 100)
        self.tr_imgsz  = spinner(32, 4096, 640, 100)
        self.tr_batch  = spinner(1, 1024, 16, 100)
        self.tr_device = QComboBox()
        self.tr_device.setMinimumWidth(100)
        self.tr_device.setProperty("themeClass", "combo")
        self.tr_device.setStyleSheet(COMBO_STYLE)
        self.tr_device.currentIndexChanged.connect(self._on_device_selected)

        grid = QHBoxLayout()
        grid.setSpacing(28)
        for lbl_text, i18n_key, wgt in [
            ("Epochs", "train.field.epochs", self.tr_epochs),
            ("Imgsz", "train.field.imgsz", self.tr_imgsz),
            ("Batch", "train.field.batch", self.tr_batch),
            ("Device", "train.field.device", self.tr_device),
        ]:
            col = QVBoxLayout()
            col.setSpacing(4)
            col.addWidget(field_label(lbl_text, i18n_key=i18n_key))
            col.addWidget(wgt)
            grid.addLayout(col)
        grid.addStretch()
        lay2.addLayout(grid)
        lay2.addSpacing(12)

        exp_row = QHBoxLayout()
        exp_row.setSpacing(10)
        exp_row.addWidget(field_label("实验名称", i18n_key="train.field.exp_name"))
        self.tr_exp = input_(min_width=320)
        exp_row.addWidget(self.tr_exp, 1)
        lay2.addLayout(exp_row)
        outer.addWidget(card2)

        # ── 训练模式卡片 ──
        card3, lay3 = card()
        lay3.addWidget(section_label("训练模式", i18n_key="train.card.mode"))
        lay3.addSpacing(12)

        self.rb_new = QRadioButton(tr("train.rb.new"))
        self.rb_new.setProperty("i18nKey", "train.rb.new")
        self.rb_new.setChecked(True)
        self.rb_new.setStyleSheet(RADIO_STYLE)
        self.rb_new.setProperty("themeClass", "radio")

        self.rb_resume = QRadioButton(tr("train.rb.resume"))
        self.rb_resume.setProperty("i18nKey", "train.rb.resume")
        self.rb_resume.setStyleSheet(RADIO_STYLE)
        self.rb_resume.setProperty("themeClass", "radio")

        self.rb_best = QRadioButton(tr("train.rb.finetune"))
        self.rb_best.setProperty("i18nKey", "train.rb.finetune")
        self.rb_best.setStyleSheet(RADIO_STYLE)
        self.rb_best.setProperty("themeClass", "radio")

        lay3.addWidget(self.rb_new)
        lay3.addWidget(self.rb_resume)
        lay3.addWidget(self.rb_best)

        hist_row = QHBoxLayout()
        hist_row.setSpacing(10)
        hist_row.addWidget(field_label("历史实验", i18n_key="train.field.history"))
        self.cb_history = simple_combo(min_width=300, font_size=13)
        hist_row.addWidget(self.cb_history, 1)
        refresh = btn("刷新", primary=False, i18n_key="train.btn.refresh")
        refresh.clicked.connect(self._refresh_history)
        hist_row.addWidget(refresh)
        lay3.addSpacing(8)
        lay3.addLayout(hist_row)
        outer.addWidget(card3)

        # ── 数据增强 ──
        self.tr_augment = QCheckBox(tr("train.augment"))
        self.tr_augment.setProperty("i18nKey", "train.augment")
        self.tr_augment.setChecked(True)
        self.tr_augment.setProperty("themeClass", "checkbox")
        self.tr_augment.setStyleSheet(CHECKBOX_STYLE)
        outer.addWidget(self.tr_augment)

        # ── 操作按钮行 ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.btn_start = btn("开始训练", i18n_key="train.btn.start")
        self.btn_start.setFixedHeight(38)
        self.btn_start.clicked.connect(self._on_start_train)
        btn_row.addWidget(self.btn_start)

        self.btn_stop = danger_btn("停止训练", i18n_key="train.btn.stop")
        self.btn_stop.setFixedHeight(38)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_train)
        btn_row.addWidget(self.btn_stop)

        self.btn_reset = btn("恢复默认", primary=False, i18n_key="train.btn.reset")
        self.btn_reset.setFixedHeight(38)
        self.btn_reset.clicked.connect(self._reset_train_defaults)
        btn_row.addWidget(self.btn_reset)

        self.cb_presets = simple_combo(min_width=120)
        self._refresh_preset_combo()
        self.cb_presets.currentTextChanged.connect(self._on_preset_selected)
        btn_row.addWidget(self.cb_presets)

        save_btn = btn("保存预设", primary=False, i18n_key="train.btn.save_preset")
        save_btn.setFixedHeight(38)
        save_btn.clicked.connect(self._save_preset)
        btn_row.addWidget(save_btn)

        del_btn = btn("删除预设", primary=False, i18n_key="train.btn.delete_preset")
        del_btn.setFixedHeight(38)
        del_btn.clicked.connect(self._delete_preset)
        btn_row.addWidget(del_btn)

        btn_row.addStretch()
        outer.addLayout(btn_row)

        # ── 进度条 ──
        outer.addSpacing(4)
        self.tr_progress = progress_bar(i18n_key="train.progress.format")
        outer.addWidget(self.tr_progress)

        # ── 日志 ──
        outer.addWidget(field_label("输出", i18n_key="train.log.output"))
        self.tr_log = log_area()
        outer.addWidget(self.tr_log, 1)

        scroll = scroll_area(w)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

        self._load_train_defaults()

    # ── public API ──

    def stop_worker(self):
        if self._train_worker and self._train_worker.isRunning():
            self._train_worker.stop()

    def is_worker_running(self) -> bool:
        return self._train_worker is not None and self._train_worker.isRunning()

    def on_ctrl_enter(self):
        self._on_start_train()

    # ── 路径历史 & 浏览 ──

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

    def _open_data_yaml(self):
        p = Path(path_combo_get(self.tr_data_yaml))
        if not p.is_file():
            QMessageBox.warning(self, tr("msg.title.hint"), f"{tr('msg.yaml_not_found')}\n{p}")
            return
        try:
            open_file_with_default_app(str(p))
        except Exception:
            QMessageBox.critical(self, tr("msg.title.error"), tr("msg.cannot_open_file"))

    # ── 日志 ──

    def _log_info(self, msg):
        log_append(self.tr_log, f'<span style="color:#6ec6ff;">{tr("log.info_prefix")}</span>  {msg}')

    def _log_good(self, msg):
        log_append(self.tr_log, f'<span style="color:#50fa7b;">{tr("log.ok_prefix")}</span>  {msg}')

    def _log_warn(self, msg):
        log_append(self.tr_log, f'<span style="color:#ffb86c;">{tr("log.warn_prefix")}</span>  {msg}')

    def _log_err(self, msg):
        log_append(self.tr_log, f'<span style="color:#ff5555;">{tr("log.err_prefix")}</span>  {msg}')

    # ── 配置管理 ──

    def _load_train_defaults(self):
        self._refresh_devices()
        self._apply_config(TrainConfig())
        self._update_dataset_info()

    def _refresh_devices(self):
        current = self.tr_device.currentData() or get_default_device()
        self.tr_device.clear()
        for dev_id, dev_name in get_available_devices():
            self.tr_device.addItem(dev_name, dev_id)
        idx = self.tr_device.findData(current)
        if idx >= 0:
            self.tr_device.setCurrentIndex(idx)
        else:
            self.tr_device.setCurrentIndex(0)

    def _on_device_selected(self, idx: int) -> None:
        device_id = self.tr_device.itemData(idx)
        if device_id != "gpu":
            return

        from gui.gpu_manager import check_gpu_capability, GpuDownloadWorker, GpuInstallWorker, get_gpu_dir

        capability = check_gpu_capability()

        if capability["status"] == "needs_download":
            reply = QMessageBox.question(
                self, tr("gpu.download_title"),
                capability["message"],
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                self.tr_device.setCurrentIndex(0)
                return

            from gui.model_selector import DownloadDialog
            self._gpu_download_dialog = DownloadDialog("CUDA components", self.window())
            self._gpu_download_dialog.show()

            RELEASE_URL = "https://github.com/Liujingze11/YOLO-LAB/releases/latest/download/gpu_bundle.zip"
            self._gpu_worker = GpuDownloadWorker(RELEASE_URL, get_gpu_dir())
            self._gpu_worker.progress.connect(self._gpu_download_dialog.set_progress)
            self._gpu_worker.finished.connect(self._on_gpu_download_done)
            self._gpu_worker.error_msg.connect(self._on_gpu_download_error)
            self._gpu_worker.start()

        elif capability["status"] == "needs_restart":
            QMessageBox.information(self, tr("gpu.check_title"), tr("gpu.restart_needed"))
            self.tr_device.setCurrentIndex(0)

        elif capability["status"] == "dev_needs_cuda_torch":
            reply = QMessageBox.question(
                self, tr("gpu.install_title"),
                capability["message"],
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                self.tr_device.setCurrentIndex(0)
                return

            from gui.model_selector import DownloadDialog
            self._gpu_install_dialog = DownloadDialog("CUDA PyTorch", self.window())
            self._gpu_install_dialog.setWindowTitle(tr("gpu.install_title"))
            self._gpu_install_dialog.show()

            self._gpu_install_worker = GpuInstallWorker()
            self._gpu_install_worker.log_line.connect(
                self._gpu_install_dialog.set_status_text
            )
            self._gpu_install_worker.finished.connect(self._on_gpu_install_done)
            self._gpu_install_worker.start()

        elif capability["status"] in ("no_nvidia", "has_amd"):
            QMessageBox.information(self, tr("gpu.check_title"), capability["message"])
            self.tr_device.setCurrentIndex(0)

    def _on_gpu_install_done(self, success: bool, msg: str) -> None:
        if self._gpu_install_dialog:
            self._gpu_install_dialog.accept()
            self._gpu_install_dialog = None
        if success:
            QMessageBox.information(self, tr("gpu.install_title"), tr("gpu.install_success"))
        else:
            QMessageBox.critical(self, tr("gpu.install_failed"), msg[:500])
        self.tr_device.setCurrentIndex(0)
        self._refresh_devices()

    def _on_gpu_download_done(self, success: bool, msg: str) -> None:
        if self._gpu_download_dialog:
            self._gpu_download_dialog.accept()
            self._gpu_download_dialog = None
        if success:
            QMessageBox.information(self, tr("gpu.download_title"), tr("gpu.restart_needed"))
        else:
            QMessageBox.critical(self, tr("gpu.download_failed"), msg[:500])
            QMessageBox.information(self, tr("gpu.download_title"), tr("gpu.install_manual"))
        self.tr_device.setCurrentIndex(0)
        self._refresh_devices()

    def _on_gpu_download_error(self, msg: str) -> None:
        if self._gpu_download_dialog:
            self._gpu_download_dialog.set_error(msg)

    def _apply_config(self, c):
        self.tr_data_yaml.setCurrentText(c.data_yaml)
        self.tr_model.set_model(c.model_file)
        self.tr_results.setCurrentText(c.results_dir)
        self.tr_logs.setCurrentText(c.log_dir)
        self.tr_epochs.setValue(int(c.epochs))
        self.tr_imgsz.setValue(int(c.imgsz))
        self.tr_batch.setValue(int(c.batch))
        idx = self.tr_device.findData(str(c.device))
        if idx >= 0:
            self.tr_device.setCurrentIndex(idx)
        else:
            self.tr_device.setCurrentIndex(0)
        self.tr_exp.setText(c.experiment_name)
        self.tr_augment.setChecked(bool(c.use_augment))
        self._refresh_history()

    def _scan_trained_models(self):
        results_dir = path_combo_get(self.tr_results)
        if not Path(results_dir).is_dir():
            return
        found = 0
        for exp in sorted(Path(results_dir).iterdir()):
            if exp.is_dir():
                best = exp / "weights" / "best.pt"
                if best.is_file():
                    self.tr_model.add_custom_path(str(best))
                    found += 1
        if found:
            self._log_info(tr("train.log.scan_found", count=found))
        else:
            self._log_warn(tr("train.log.scan_none"))

    def _reset_train_defaults(self):
        self._apply_config(TrainConfig())
        self._log_info(tr("train.log.defaults_reset"))

    def _refresh_history(self):
        self.cb_history.clear()
        res = Path(path_combo_get(self.tr_results) or ".")
        if not res.is_dir():
            return
        for name in sorted(list_experiments(str(res))):
            self.cb_history.addItem(name)

    # ── 预设 ──

    def _get_current_config_dict(self):
        return {
            "data_yaml": path_combo_get(self.tr_data_yaml),
            "model_file": self.tr_model.current_model_path(),
            "results_dir": path_combo_get(self.tr_results),
            "log_dir": path_combo_get(self.tr_logs),
            "epochs": self.tr_epochs.value(),
            "imgsz": self.tr_imgsz.value(),
            "batch": self.tr_batch.value(),
            "device": self.tr_device.currentData() or get_default_device(),
            "experiment_name": self.tr_exp.text().strip(),
            "use_augment": self.tr_augment.isChecked(),
        }

    def _apply_config_dict(self, d):
        self.tr_data_yaml.setCurrentText(d.get("data_yaml", ""))
        self.tr_model.set_model(d.get("model_file", ""))
        self.tr_results.setCurrentText(d.get("results_dir", ""))
        self.tr_logs.setCurrentText(d.get("log_dir", ""))
        self.tr_epochs.setValue(d.get("epochs", 150))
        self.tr_imgsz.setValue(d.get("imgsz", 640))
        self.tr_batch.setValue(d.get("batch", 16))
        dev = d.get("device", get_default_device())
        idx = self.tr_device.findData(dev)
        if idx >= 0:
            self.tr_device.setCurrentIndex(idx)
        else:
            self.tr_device.setCurrentIndex(0)
        self.tr_exp.setText(d.get("experiment_name", ""))
        self.tr_augment.setChecked(d.get("use_augment", True))
        self._refresh_history()

    def _refresh_preset_combo(self):
        self.cb_presets.blockSignals(True)
        self.cb_presets.clear()
        self.cb_presets.addItem(tr("train.combo.presets"))
        self._presets = load_presets()
        for name in sorted(self._presets.keys()):
            self.cb_presets.addItem(name)
        self.cb_presets.blockSignals(False)

    def _on_preset_selected(self, name):
        placeholder = tr("train.combo.presets")
        if not name or name == placeholder or name not in self._presets:
            return
        self._apply_config_dict(self._presets[name])
        self._log_info(tr("train.log.preset_loaded", name=name))

    def _save_preset(self):
        name = self.tr_exp.text().strip()
        if not name:
            name = "default"
        self._presets[name] = self._get_current_config_dict()
        save_presets(self._presets)
        self._refresh_preset_combo()
        idx = self.cb_presets.findText(name)
        if idx >= 0:
            self.cb_presets.setCurrentIndex(idx)
        self._log_info(tr("train.log.preset_saved", name=name))

    def _delete_preset(self):
        name = self.cb_presets.currentText()
        placeholder = tr("train.combo.presets")
        if not name or name == placeholder:
            QMessageBox.warning(self, tr("msg.title.hint"), tr("msg.select_experiment"))
            return
        if name in self._presets:
            del self._presets[name]
            save_presets(self._presets)
            self._refresh_preset_combo()
            self._log_info(tr("train.log.preset_deleted", name=name))

    # ── 训练 ──

    def _set_train_ui_state(self, state: str) -> None:
        if state == "running":
            self.btn_start.setEnabled(False)
            self.btn_stop.setText(tr("train.btn.stop"))
            self.btn_stop.setProperty("i18nKey", "train.btn.stop")
            self.btn_stop.setEnabled(True)
            self.btn_stop.clicked.disconnect()
            self.btn_stop.clicked.connect(self._on_stop_train)
            self.tr_progress.setValue(0)
        elif state == "stopped":
            self.btn_start.setText(tr("train.btn.continue"))
            self.btn_start.setProperty("i18nKey", "train.btn.continue")
            self.btn_start.setEnabled(True)
            self.btn_stop.setText(tr("train.btn.end"))
            self.btn_stop.setProperty("i18nKey", "train.btn.end")
            self.btn_stop.setEnabled(True)
            self.rb_resume.setChecked(True)
            self.btn_stop.clicked.disconnect()
            self.btn_stop.clicked.connect(self._on_end_train)
        else:  # idle
            self.btn_start.setText(tr("train.btn.start"))
            self.btn_start.setProperty("i18nKey", "train.btn.start")
            self.btn_start.setEnabled(True)
            self.btn_stop.setText(tr("train.btn.stop"))
            self.btn_stop.setProperty("i18nKey", "train.btn.stop")
            self.btn_stop.setEnabled(False)
            self.rb_new.setChecked(True)
            self.btn_stop.clicked.disconnect()
            self.btn_stop.clicked.connect(self._on_stop_train)
            self.tr_progress.setRange(0, 100)
            self.tr_progress.setValue(0)
            self.tr_progress.setFormat(tr("train.progress.format"))

    def _update_dataset_info(self):
        p = Path(path_combo_get(self.tr_data_yaml))
        if not p.is_file():
            self.dataset_info.setText("")
            return
        try:
            import yaml
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            names = data.get("names", {})
            nc = len(names) if isinstance(names, (dict, list)) else 0
            root = data.get("path", "")
            if root and not Path(root).is_absolute():
                root = str((p.parent / root).resolve())
            elif root:
                root = str(Path(root).resolve())
            parts = [f"nc={nc}"]
            if root:
                parts.append(root)
            self.dataset_info.setText("  ".join(parts))
        except Exception:
            self.dataset_info.setText("")

    @staticmethod
    def _validate_config(cfg) -> list[str]:
        errors = []
        # data.yaml
        if not cfg.data_yaml:
            errors.append(tr("validate.no_data_yaml"))
        elif not Path(cfg.data_yaml).is_file():
            errors.append(f"{tr('validate.data_yaml_missing')}\n  {cfg.data_yaml}")
        # model file
        if not model_file_ok(cfg.model_file):
            errors.append(f"{tr('validate.model_missing')}\n  {cfg.model_file}")
        # results dir
        if not cfg.results_dir:
            errors.append(tr("validate.no_results_dir"))
        # experiment name
        if not cfg.experiment_name.strip():
            errors.append(tr("validate.no_exp_name"))
        return errors

    def _build_config_from_train_ui(self):
        c = TrainConfig()
        c.data_yaml = path_combo_get(self.tr_data_yaml)
        c.model_file = self.tr_model.current_model_path()
        c.results_dir = path_combo_get(self.tr_results)
        c.log_dir = path_combo_get(self.tr_logs)
        c.epochs = int(self.tr_epochs.value())
        c.imgsz = int(self.tr_imgsz.value())
        c.batch = int(self.tr_batch.value())
        c.device = self.tr_device.currentData() or get_default_device()
        c.experiment_name = self.tr_exp.text().strip() or c.experiment_name
        c.use_augment = self.tr_augment.isChecked()
        return c

    @Slot()
    def _on_start_train(self):
        if self._train_worker and self._train_worker.isRunning():
            QMessageBox.warning(self, tr("msg.title.hint"), tr("msg.train_running"))
            return

        cfg = self._build_config_from_train_ui()
        use_aug = self.tr_augment.isChecked()

        # Validate config before proceeding
        errors = self._validate_config(cfg)
        if errors:
            QMessageBox.critical(self, tr("msg.title.error"), "\n".join(errors))
            return

        if self.rb_new.isChecked():
            mode = 1
            mode_label = tr("train.msg.mode_new")
            if not model_file_ok(cfg.model_file):
                QMessageBox.critical(self, tr("msg.title.error"), f"{tr('msg.no_model')}\n{cfg.model_file}")
                return
            selected = None
            details = tr("train.msg.new_summary",
                         mode=mode_label, exp=cfg.experiment_name,
                         weights=cfg.model_file, data=cfg.data_yaml)
        elif self.rb_resume.isChecked():
            mode = 2
            mode_label = tr("train.msg.mode_resume")
            if not Path(cfg.last_pt).is_file():
                r = QMessageBox.question(
                    self, tr("msg.title.hint"),
                    f"{tr('msg.no_last_pt')}\n{cfg.last_pt}\n\n{tr('msg.fallback_new_train')}",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if r != QMessageBox.Yes:
                    return
                mode = 1
                mode_label = tr("train.msg.mode_new_fallback")
                if not model_file_ok(cfg.model_file):
                    QMessageBox.critical(self, tr("msg.title.error"), f"{tr('msg.no_model')}\n{cfg.model_file}")
                    return
            selected = None
            weights_path = cfg.last_pt if mode == 2 else cfg.model_file
            details = tr("train.msg.resume_summary",
                         mode=mode_label, exp=cfg.experiment_name,
                         weights=weights_path, data=cfg.data_yaml)
        else:
            mode = 3
            mode_label = tr("train.msg.mode_finetune")
            selected = self.cb_history.currentText().strip()
            if not selected:
                QMessageBox.warning(self, tr("msg.title.hint"), tr("msg.no_history_selected"))
                return
            best = Path(cfg.results_dir) / selected / "weights" / "best.pt"
            if not best.is_file():
                QMessageBox.critical(self, tr("msg.title.error"), f"{tr('msg.no_best_pt')}\n{best}")
                return
            details = tr("train.msg.finetune_summary",
                         mode=mode_label, exp=cfg.experiment_name,
                         base=selected, weights=best, data=cfg.data_yaml)

        aug_status = tr("train.engine.augment_on") if use_aug else tr("train.engine.augment_off")
        summary = (
            f"{details}\n"
            f"{'─' * 40}\n"
            f"{tr('train.field.epochs')}:  {cfg.epochs:<6}  {tr('train.field.imgsz')}: {cfg.imgsz}\n"
            f"{tr('train.field.batch')}:   {cfg.batch:<6}  {tr('train.field.device')}: {cfg.device}\n"
            f"{aug_status}\n"
            f"{'─' * 40}\n"
            f"{tr('msg.title.confirm')}"
        )

        r = QMessageBox.question(
            self, tr("msg.title.confirm"), summary,
            QMessageBox.Yes | QMessageBox.No,
        )
        if r != QMessageBox.Yes:
            return

        self.tr_log.clear()
        self._log_info(tr("train.log.starting", name=cfg.experiment_name))
        self._log_info(tr("train.log.params", epochs=cfg.epochs, imgsz=cfg.imgsz, batch=cfg.batch, device=cfg.device))

        cmd = engine_cmd("--engine-train") + [
            "--lang", current_lang(),
            "--no-interactive",
            "--mode", str(mode),
            "--data-yaml", cfg.data_yaml,
            "--model-file", cfg.model_file,
            "--results-dir", cfg.results_dir,
            "--log-dir", cfg.log_dir,
            "--epochs", str(cfg.epochs),
            "--imgsz", str(cfg.imgsz),
            "--batch", str(cfg.batch),
            "--device", cfg.device,
            "--name", cfg.experiment_name,
        ]
        if use_aug:
            cmd.append("--use-augment")
        else:
            cmd.append("--no-augment")
        if mode == 3 and selected:
            cmd.extend(["--selected-exp", selected])

        self._set_train_ui_state("running")
        self.tr_progress.setRange(0, cfg.epochs)
        self.tr_progress.setValue(0)
        self.tr_progress.setFormat(tr("train.progress.format"))

        self._train_worker = TrainWorker(cmd)
        self._train_worker.log_line.connect(self._append_train_log)
        self._train_worker.progress.connect(self._on_train_progress)
        self._train_worker.failed.connect(self._on_train_failed)
        self._train_worker.finished_ok.connect(self._on_train_done)
        self._train_worker.stopped.connect(self._on_train_stopped)
        self._train_worker.finished.connect(self._on_train_thread_finished)
        self._train_worker.start()

    @Slot(str)
    def _append_train_log(self, line):
        log_append(self.tr_log, f'<span style="color:#c0c0c0;">{line}</span>')

    @Slot(int)
    def _on_train_progress(self, pct: int) -> None:
        self.tr_progress.setValue(pct)

    @Slot()
    def _on_stop_train(self):
        if self._train_worker and self._train_worker.isRunning():
            self._log_warn(tr("train.log.stopping"))
            self._train_worker.stop()

    @Slot(str)
    def _on_train_failed(self, msg):
        self._log_err(tr("msg.title.failed"))
        log_append(self.tr_log, f'<span style="color:#ff6e6e;">{msg[:1500]}</span>')
        QMessageBox.critical(self, tr("msg.title.failed"), msg[:2000])
        self._set_train_ui_state("idle")
        self._refresh_history()

    @Slot()
    def _on_train_done(self):
        self._log_good(tr("msg.train_done"))
        QMessageBox.information(self, tr("msg.title.done"), tr("msg.train_done"))
        self._set_train_ui_state("idle")
        self._refresh_history()

    @Slot()
    def _on_train_stopped(self):
        self._log_warn(tr("train.log.paused"))
        self._set_train_ui_state("stopped")

    @Slot()
    def _on_end_train(self):
        self._log_info(tr("train.log.ended"))
        self._set_train_ui_state("idle")

    @Slot()
    def _on_train_thread_finished(self):
        self.closing.emit()
