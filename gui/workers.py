"""
后台工作线程 —— 在子进程中运行训练/推理/工具脚本，实时读取输出。
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from PySide6.QtCore import QThread, Signal

ROOT = Path(__file__).resolve().parent


class _BaseWorker(QThread):
    """子进程工作线程基类。"""

    log_line = Signal(str)
    finished_ok = Signal()
    failed = Signal(str)
    stopped = Signal()

    def __init__(self, cmd: list[str]):
        super().__init__()
        self._cmd = cmd
        self._process: subprocess.Popen | None = None
        self._aborted = False

    def run(self) -> None:
        try:
            self._process = subprocess.Popen(
                self._cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(ROOT),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except Exception as e:
            self.failed.emit(f"启动进程失败: {e}")
            return

        try:
            for line in self._process.stdout:
                stripped = line.rstrip("\n").rstrip("\r")
                if stripped:
                    self.log_line.emit(stripped)
                    self._on_line(stripped)
        except Exception as e:
            # stdout read error (e.g. encoding issue) — not fatal, report what we can
            self.log_line.emit(f"[worker] stdout read error: {e}")

        self._process.wait()
        if self._aborted:
            self.stopped.emit()
        elif self._process.returncode == 0:
            self.finished_ok.emit()
        else:
            self.failed.emit(f"进程退出码: {self._process.returncode}")

    def _on_line(self, line: str) -> None:
        """子类可覆盖此方法来解析进度等。"""

    def stop(self) -> None:
        self._aborted = True
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()


class TrainWorker(_BaseWorker):
    """在子进程中运行训练脚本，解析 epoch 进度和训练指标。"""

    progress = Signal(int)
    metrics = Signal(dict)  # {epoch, box_loss, seg_loss, cls_loss, dfl_loss, lr}

    # Matches ultralytics per-epoch metrics row:
    #   epoch/total   mem   box_loss   seg_loss   cls_loss   dfl_loss   instances   size
    _METRIC_RE = re.compile(
        r"\s*(\d+)\s*/\s*(\d+)\s+"       # epoch/total
        r"(\S+)\s+"                        # GPU memory
        r"([\d.]+)\s+"                     # box_loss
        r"([\d.]+)\s+"                     # seg_loss
        r"([\d.]+)\s+"                     # cls_loss
        r"([\d.]+)\s+"                     # dfl_loss
        r"(\d+)\s+"                        # instances
        r"(\d+)"                           # size
    )

    def _on_line(self, line: str) -> None:
        # Try structured metrics first
        mm = self._METRIC_RE.search(line)
        if mm:
            cur, total = int(mm.group(1)), int(mm.group(2))
            if 1 <= cur <= total and total >= 10:
                self.progress.emit(cur)
                self.metrics.emit({
                    "epoch": cur,
                    "total": total,
                    "box_loss": float(mm.group(4)),
                    "seg_loss": float(mm.group(5)),
                    "cls_loss": float(mm.group(6)),
                    "dfl_loss": float(mm.group(7)),
                })
            return

        # Fallback: simple epoch detection
        m = re.search(r"\b(\d+)\s*/\s*(\d+)\b", line)
        if not m:
            return
        cur, total = int(m.group(1)), int(m.group(2))
        low = line.lower()
        if 1 <= cur <= total and total >= 10 and not any(
            kw in low
            for kw in (
                "transfer", "gflops", "summary", "param", "module",
                "cuda", "gradient", "amp", "fuse",
            )
        ):
            self.progress.emit(cur)
            self.metrics.emit({"epoch": cur, "total": total})


class InferWorker(_BaseWorker):
    """在子进程中运行推理脚本，解析图片进度。"""

    progress = Signal(int, int)  # cur, total

    def _on_line(self, line: str) -> None:
        low = line.lower()
        if "image" not in low:
            return
        m = re.search(r"\b(\d+)\s*/\s*(\d+)\b", line)
        if not m:
            return
        cur, total = int(m.group(1)), int(m.group(2))
        if 1 <= cur <= total:
            self.progress.emit(cur, total)


class ToolWorker(_BaseWorker):
    """在子进程中运行数据集工具脚本，只输出日志。"""
