"""
Shared utility functions used across the GUI.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor

from paths import get_preset_file, is_frozen
from gui.i18n import tr

MAX_LOG_LINES = 5000

_PRESET_FILE = get_preset_file()


def load_presets() -> dict:
    if _PRESET_FILE.is_file():
        try:
            return json.loads(_PRESET_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_presets(presets: dict) -> None:
    _PRESET_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PRESET_FILE.write_text(
        json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def log_append(log_widget, html: str, max_lines: int = MAX_LOG_LINES) -> None:
    log_widget.append(html)
    doc = log_widget.document()
    excess = doc.blockCount() - max_lines
    if excess > 0:
        cursor = log_widget.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        for _ in range(excess):
            cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
        log_widget.moveCursor(QTextCursor.MoveOperation.End)


def model_file_ok(path: str) -> bool:
    if Path(path).is_file():
        return True
    if path and os.sep not in path and "/" not in path:
        from paths import PRETRAINED_DIR
        return (PRETRAINED_DIR / path).is_file()
    return False


def open_file_with_default_app(path: str) -> None:
    import platform
    system = platform.system()
    if system == "Windows":
        os.startfile(path)
    elif system == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def open_dir_safe(path_str: str) -> None:
    p = Path(path_str)
    if p.is_dir():
        open_file_with_default_app(str(p))
    elif p.parent.is_dir():
        open_file_with_default_app(str(p.parent))


def load_csv_log(log_widget, csv_path: Path):
    log_widget.clear()
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        log_append(log_widget,
            f'<span style="color:#6ec6ff;">{tr("log.info_prefix")}</span>  {tr("msg.log_empty")}')
        return
    log_append(log_widget,
        f'<span style="color:#6ec6ff;">{tr("log.info_prefix")}</span>  {tr("msg.log_loaded")}: {csv_path.name}  ({len(rows)} {tr("msg.log_rows")})')
    html = '<table style="font-size:11px; border-collapse:collapse; width:100%;">'
    for i, row in enumerate(rows):
        tag = "th" if i == 0 else "td"
        color = "#8ab4f8" if i == 0 else "#c0c0c0"
        html += f'<tr style="color:{color};">'
        for cell in row:
            html += f"<{tag} style='padding:2px 8px; border-bottom:1px solid #333;'>{cell}</{tag}>"
        html += "</tr>"
    html += "</table>"
    log_append(log_widget, html)


def engine_cmd(engine_flag: str) -> list[str]:
    """Build subprocess command prefix that works in both dev and frozen modes."""
    import sys
    cmd = [sys.executable]
    if not is_frozen():
        cmd.append(str(Path(sys.argv[0]).resolve()))
    cmd.append(engine_flag)
    return cmd
