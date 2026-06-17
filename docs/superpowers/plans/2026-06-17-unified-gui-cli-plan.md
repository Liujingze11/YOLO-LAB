# 统一 GUI/CLI 架构迁移 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 YOLO-LAB 重构为按系统分文件夹的独立架构，`linux/` 目录完全自包含，统一入口支持 GUI/CLI 双模式。

**Architecture:** 每系统文件夹 (`linux/`, `windows/`, `macos/`) 是完全独立的应用。`linux/entry.py` 是唯一入口 — 无参启动 GUI，有子命令走 CLI。CLI 四层子命令 (train/infer/tools/config)，配置文件 ~/.yolo-lab/config.yaml 三层优先级。

**Tech Stack:** Python 3.10+, PySide6, ultralytics, PyYAML, argparse, PyInstaller

## Global Constraints

- linux/ 目录完全自包含，不依赖 shared/、gui/、cli/ 旧目录
- 入口逻辑：无参数 → GUI，有子命令 → CLI
- CLI 参数 > ~/.yolo-lab/config.yaml > 代码默认值
- 先做 Linux，windows/ 和 macos/ 只创建空壳占位
- 迁移 = 搬文件 + 修导入路径，不重写业务逻辑
- 回退点：commit `95e392c`

---

### Task 1: 创建 linux/ 目录骨架 + entry.py + config.py

**Files:**
- Create: `linux/__init__.py`
- Create: `linux/entry.py`
- Create: `linux/config.py`
- Create: `linux/paths.py`
- Create: `linux/data.yaml`
- Create: `linux/requirements.txt`
- Create: `linux/assets/icon.png`
- Modify: `packaging/assets/icon.png` (copy to linux/assets/)

**Interfaces:**
- Produces:
  - `linux/entry.py`: `main()` — 入口函数，无参→GUI，有参→CLI
  - `linux/config.py`: `TrainConfig` dataclass, `load_effective_config(cli_args) -> TrainConfig`, `load_user_config() -> dict | None`, `merge_config(cfg, overrides) -> TrainConfig`, `save_user_config(cfg: TrainConfig)`, `USER_CONFIG_DIR: Path`
  - `linux/paths.py`: `is_frozen() -> bool`, `get_app_root() -> Path`, `get_user_data_dir() -> Path`, `ensure_user_dirs() -> dict`, `get_preset_file() -> Path`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p linux/assets linux/gui/tabs linux/gui/tools/dataset_tools/split_train_val linux/gui/tools/dataset_tools/split_train_val_test linux/gui/tools/dataset_tools/split_images_only linux/locales
```

- [ ] **Step 2: 编写 `linux/__init__.py`**

```python
"""YOLO-LAB Linux — YOLO Segmentation Training & Inference Toolkit."""
```

- [ ] **Step 3: 编写 `linux/paths.py`**

```python
"""Application paths — works in dev and frozen (PyInstaller) modes."""
from pathlib import Path
import os
import sys
import platform


def is_frozen() -> bool:
    """True when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def get_app_root() -> Path:
    """Return the application root directory."""
    if is_frozen():
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def get_user_data_dir() -> Path:
    """Platform-standard user data directory."""
    app_name = "YoloLab"
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / app_name


def ensure_user_dirs() -> dict:
    """Create and return user data subdirectories."""
    data_dir = get_user_data_dir()
    dirs = {
        "models": data_dir / "models",
        "results": data_dir / "results",
        "logs": data_dir / "logs",
        "predict": data_dir / "predict",
        "gpu": data_dir / "gpu",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def get_preset_file() -> Path:
    """Return the path to presets.json."""
    if is_frozen():
        return get_user_data_dir() / "presets.json"
    return get_app_root() / "presets.json"


# ── Default paths (from gui/paths.py) ──
APP_ROOT = get_app_root()
DATA_YAML = str(APP_ROOT / "data.yaml")
MODEL_FILE = "yolov8n-seg.pt"
RESULTS_DIR = str(APP_ROOT / "outputs" / "results")
LOG_DIR = str(APP_ROOT / "outputs" / "logs")

PRETRAINED_DIR = APP_ROOT / "pretrained_models"

_MODEL_URL_BASE = "https://github.com/ultralytics/assets/releases/download"
_MODEL_URL_TAGS = {
    "v8": "v8.2.0",
    "v11": "v8.3.0",
    "v12": "v8.3.0",
}

MODEL_REGISTRY: list[tuple[str, str, str]] = [
    ("yolov8n-seg.pt",  "YOLOv8n-seg",  "v8"),
    ("yolov8s-seg.pt",  "YOLOv8s-seg",  "v8"),
    ("yolov8m-seg.pt",  "YOLOv8m-seg",  "v8"),
    ("yolov8l-seg.pt",  "YOLOv8l-seg",  "v8"),
    ("yolov8x-seg.pt",  "YOLOv8x-seg",  "v8"),
    ("yolo11n-seg.pt",  "YOLO11n-seg",  "v11"),
    ("yolo11s-seg.pt",  "YOLO11s-seg",  "v11"),
    ("yolo11m-seg.pt",  "YOLO11m-seg",  "v11"),
    ("yolo11l-seg.pt",  "YOLO11l-seg",  "v11"),
    ("yolo11x-seg.pt",  "YOLO11x-seg",  "v11"),
    ("yolo12n-seg.pt",  "YOLO12n-seg",  "v12"),
    ("yolo12s-seg.pt",  "YOLO12s-seg",  "v12"),
    ("yolo12m-seg.pt",  "YOLO12m-seg",  "v12"),
    ("yolo12l-seg.pt",  "YOLO12l-seg",  "v12"),
    ("yolo12x-seg.pt",  "YOLO12x-seg",  "v12"),
    ("yolov8n.pt",  "YOLOv8n",   "v8"),
    ("yolov8s.pt",  "YOLOv8s",   "v8"),
    ("yolov8m.pt",  "YOLOv8m",   "v8"),
    ("yolov8l.pt",  "YOLOv8l",   "v8"),
    ("yolov8x.pt",  "YOLOv8x",   "v8"),
    ("yolo11n.pt",  "YOLO11n",  "v11"),
    ("yolo11s.pt",  "YOLO11s",  "v11"),
    ("yolo11m.pt",  "YOLO11m",  "v11"),
    ("yolo11l.pt",  "YOLO11l",  "v11"),
    ("yolo11x.pt",  "YOLO11x",  "v11"),
]


def get_model_download_url(filename: str, tag_key: str) -> str:
    tag = _MODEL_URL_TAGS.get(tag_key, "v8.3.0")
    return f"{_MODEL_URL_BASE}/{tag}/{filename}"


PREDICT_DIR = str(APP_ROOT / "outputs" / "predict")
BEST_SEG_MODEL = ""
TEST_IMAGES_DIR = str(APP_ROOT / "data" / "dataset" / "images" / "test")
```

- [ ] **Step 4: 编写 `linux/config.py`**

```python
"""Unified TrainConfig + user config file management."""
import os
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path


def _default_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


@dataclass
class TrainConfig:
    data_yaml: str = ""
    model_file: str = ""
    results_dir: str = ""
    log_dir: str = ""

    epochs: int = 150
    imgsz: int = 640
    batch: int = 16
    device: str = field(default_factory=_default_device)

    experiment_name: str = "experiment"

    use_augment: bool = True
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0

    @property
    def save_dir(self) -> str:
        return os.path.join(self.results_dir, self.experiment_name)

    @property
    def last_pt(self) -> str:
        return os.path.join(self.save_dir, "weights", "last.pt")

    @property
    def best_pt(self) -> str:
        return os.path.join(self.save_dir, "weights", "best.pt")


# ── User config file (~/.yolo-lab/config.yaml) ──

_USER_CONFIG_DIR = Path.home() / ".yolo-lab"
_USER_CONFIG_PATH = _USER_CONFIG_DIR / "config.yaml"


def load_user_config() -> dict | None:
    """Load user config from ~/.yolo-lab/config.yaml. Returns None if not found."""
    if not _USER_CONFIG_PATH.is_file():
        return None
    try:
        with open(_USER_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def save_user_config(cfg: TrainConfig) -> None:
    """Save TrainConfig to ~/.yolo-lab/config.yaml."""
    _USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    d = {k: v for k, v in asdict(cfg).items() if not k.startswith("_")}
    with open(_USER_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(d, f, allow_unicode=True, default_flow_style=False)


def merge_config(base: TrainConfig, overrides: dict) -> TrainConfig:
    """Apply overrides dict onto a TrainConfig, returning a NEW TrainConfig."""
    d = asdict(base)
    for k, v in overrides.items():
        if v is not None and k in d:
            d[k] = v
    return TrainConfig(**d)


def load_effective_config(cli_args: dict | None = None) -> TrainConfig:
    """Merge: defaults → ~/.yolo-lab/config.yaml → CLI args."""
    cfg = TrainConfig()
    file_cfg = load_user_config()
    if file_cfg:
        cfg = merge_config(cfg, file_cfg)
    if cli_args:
        cfg = merge_config(cfg, cli_args)
    return cfg
```

- [ ] **Step 5: 编写 `linux/data.yaml`**（从 `cli/data.yaml` 复制）

```bash
cp cli/data.yaml linux/data.yaml
```

- [ ] **Step 6: 编写 `linux/requirements.txt`**

```bash
# 检查是否有 gui/requirements.txt，如有则复制；否则创建
if [ -f gui/requirements.txt ]; then
    cp gui/requirements.txt linux/requirements.txt
else
    cat > linux/requirements.txt << 'DEPS'
PySide6>=6.5
torch>=2.0
ultralytics>=8.0
numpy>=1.24
opencv-python>=4.8
matplotlib>=3.7
pyyaml>=6.0
pillow>=10.0
DEPS
fi
```

- [ ] **Step 7: 复制图标**

```bash
cp packaging/assets/icon.png linux/assets/icon.png
```

- [ ] **Step 8: 编写 `linux/entry.py`**

```python
"""YOLO-LAB unified entry point — no-arg → GUI, subcommand → CLI."""
import sys
from pathlib import Path

# Ensure the linux/ directory is on sys.path for all imports
LINUX_ROOT = Path(__file__).resolve().parent
if str(LINUX_ROOT) not in sys.path:
    sys.path.insert(0, str(LINUX_ROOT))


def main():
    if len(sys.argv) > 1:
        # Has subcommand → CLI mode
        from cli import run_cli
        run_cli(sys.argv[1:])
    else:
        # No arguments → GUI mode
        from gui.main_window import run_gui
        run_gui()


if __name__ == "__main__":
    main()
```

- [ ] **Step 9: 验证骨架可 import**

Run: `cd linux && python -c "from paths import APP_ROOT; from config import TrainConfig; print('OK')"`
Expected: `OK`

- [ ] **Step 10: Commit**

```bash
git add linux/__init__.py linux/entry.py linux/config.py linux/paths.py linux/data.yaml linux/requirements.txt linux/assets/
git commit -m "feat: create linux/ skeleton — entry, config, paths"
```

---

### Task 2: 迁移训练核心逻辑到 linux/

**Files:**
- Create: `linux/train_logger.py`
- Create: `linux/training.py`
- Create: `linux/i18n.py`

**Interfaces:**
- Consumes: `linux/config.py` (TrainConfig)
- Produces:
  - `linux/train_logger.py`: `get_timestamp()`, `ensure_log_dir(log_dir)`, `append_train_log(config, mode, status, notes)`, `append_full_val_log(config, mode, metrics, ...)`, `extract_seg_val_metrics(metrics, ...)`
  - `linux/training.py`: `list_experiments(results_dir) -> list`, `build_train_kwargs(config, use_augment) -> dict`, `get_class_names_from_data_yaml(data_yaml_path) -> dict`, `get_val_labels_dir(data_yaml_path) -> str | None`, `count_val_label_stats(config) -> tuple`, `get_val_metrics(best_pt_path, config)`
  - `linux/i18n.py`: `load_locale(locale_dir, lang) -> dict`, `t(loc, key, **kwargs) -> str`

- [ ] **Step 1: 编写 `linux/train_logger.py`** — 从 `shared/train_logger.py` 复制，修改 import

```python
"""Training log CSV management."""
import os
import csv
from datetime import datetime


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_log_dir(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)


# ── train_log.csv ──

def ensure_train_csv_header(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time", "mode", "status", "experiment_name", "model_file",
                "data_yaml", "epochs", "imgsz", "batch", "device",
                "save_dir", "best_pt", "last_pt", "notes",
            ])


def append_train_log(config, mode: str, status: str, notes: str = ""):
    ensure_log_dir(config.log_dir)
    csv_path = os.path.join(config.log_dir, "train_log.csv")
    ensure_train_csv_header(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            get_timestamp(), mode, status, config.experiment_name,
            config.model_file, config.data_yaml,
            config.epochs, config.imgsz, config.batch, config.device,
            config.save_dir, config.best_pt, config.last_pt, notes,
        ])


# ── result_summary_log.csv ──

def ensure_result_summary_csv_header(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time", "mode", "experiment_name", "best_pt",
                "images", "instances",
                "box_p", "box_r", "box_map50", "box_map50_95",
                "mask_p", "mask_r", "mask_map50", "mask_map50_95",
                "notes",
            ])


def append_result_summary_log(config, mode: str, summary: dict, notes: str = ""):
    ensure_log_dir(config.log_dir)
    csv_path = os.path.join(config.log_dir, "result_summary_log.csv")
    ensure_result_summary_csv_header(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            get_timestamp(), mode, config.experiment_name, config.best_pt,
            summary.get("images", ""), summary.get("instances", ""),
            summary.get("box_p", ""), summary.get("box_r", ""),
            summary.get("box_map50", ""), summary.get("box_map50_95", ""),
            summary.get("mask_p", ""), summary.get("mask_r", ""),
            summary.get("mask_map50", ""), summary.get("mask_map50_95", ""),
            notes,
        ])


# ── result_per_class_log.csv ──

def ensure_result_per_class_csv_header(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time", "mode", "experiment_name", "best_pt",
                "class_id", "class_name", "images", "instances",
                "box_p", "box_r", "box_map50", "box_map50_95",
                "mask_p", "mask_r", "mask_map50", "mask_map50_95",
                "notes",
            ])


def append_result_per_class_log(config, mode: str, class_rows: list, notes: str = ""):
    ensure_log_dir(config.log_dir)
    csv_path = os.path.join(config.log_dir, "result_per_class_log.csv")
    ensure_result_per_class_csv_header(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for row in class_rows:
            writer.writerow([
                get_timestamp(), mode, config.experiment_name, config.best_pt,
                row.get("class_id", ""), row.get("class_name", ""),
                row.get("images", ""), row.get("instances", ""),
                row.get("box_p", ""), row.get("box_r", ""),
                row.get("box_map50", ""), row.get("box_map50_95", ""),
                row.get("mask_p", ""), row.get("mask_r", ""),
                row.get("mask_map50", ""), row.get("mask_map50_95", ""),
                notes,
            ])


# ── Metrics extraction ──

def extract_seg_val_metrics(metrics, class_image_counts=None, class_instance_counts=None):
    class_image_counts = class_image_counts or {}
    class_instance_counts = class_instance_counts or {}

    mean_vals = metrics.mean_results()

    summary = {
        "images": sum(class_image_counts.values()) if class_image_counts else "",
        "instances": sum(class_instance_counts.values()) if class_instance_counts else "",
        "box_p": mean_vals[0] if len(mean_vals) > 0 else "",
        "box_r": mean_vals[1] if len(mean_vals) > 1 else "",
        "box_map50": mean_vals[2] if len(mean_vals) > 2 else "",
        "box_map50_95": mean_vals[3] if len(mean_vals) > 3 else "",
        "mask_p": mean_vals[4] if len(mean_vals) > 4 else "",
        "mask_r": mean_vals[5] if len(mean_vals) > 5 else "",
        "mask_map50": mean_vals[6] if len(mean_vals) > 6 else "",
        "mask_map50_95": mean_vals[7] if len(mean_vals) > 7 else "",
    }

    per_class_rows = []
    names = metrics.names or {}

    def _is_valid_class_name(cname):
        if cname is None:
            return False
        return str(cname).strip().lower() not in {"none", "", "background", "__background__"}

    valid_classes = [(cid, cname) for cid, cname in names.items() if _is_valid_class_name(cname)]

    for idx, (class_id, class_name) in enumerate(valid_classes):
        try:
            vals = metrics.class_result(idx)
        except Exception:
            vals = []

        row = {
            "class_id": class_id,
            "class_name": class_name,
            "images": class_image_counts.get(class_name, 0),
            "instances": class_instance_counts.get(class_name, 0),
            "box_p": vals[0] if len(vals) > 0 else "",
            "box_r": vals[1] if len(vals) > 1 else "",
            "box_map50": vals[2] if len(vals) > 2 else "",
            "box_map50_95": vals[3] if len(vals) > 3 else "",
            "mask_p": vals[4] if len(vals) > 4 else "",
            "mask_r": vals[5] if len(vals) > 5 else "",
            "mask_map50": vals[6] if len(vals) > 6 else "",
            "mask_map50_95": vals[7] if len(vals) > 7 else "",
        }
        per_class_rows.append(row)

    return summary, per_class_rows


def append_full_val_log(config, mode: str, metrics, class_image_counts=None,
                        class_instance_counts=None, notes: str = ""):
    summary, per_class_rows = extract_seg_val_metrics(
        metrics,
        class_image_counts=class_image_counts,
        class_instance_counts=class_instance_counts,
    )
    append_result_summary_log(config, mode, summary, notes)
    append_result_per_class_log(config, mode, per_class_rows, notes)
```

- [ ] **Step 2: 编写 `linux/training.py`** — 从 `shared/train_core.py` 复制，修改 import

```python
"""Shared training/validation logic."""
import os
import yaml
import shutil
from pathlib import Path


def list_experiments(results_dir: str) -> list:
    if not os.path.exists(results_dir):
        return []
    return sorted(
        name for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name))
    )


def build_train_kwargs(config, use_augment: bool) -> dict:
    kwargs = {
        "data": config.data_yaml,
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "device": config.device,
        "project": config.results_dir,
        "name": config.experiment_name,
        "exist_ok": True,
    }
    if use_augment:
        kwargs.update({
            "hsv_h": config.hsv_h, "hsv_s": config.hsv_s, "hsv_v": config.hsv_v,
            "degrees": config.degrees, "translate": config.translate,
            "scale": config.scale, "shear": config.shear,
            "perspective": config.perspective, "flipud": config.flipud,
            "fliplr": config.fliplr, "mosaic": config.mosaic,
            "mixup": config.mixup, "copy_paste": config.copy_paste,
        })
    return kwargs


def get_class_names_from_data_yaml(data_yaml_path: str) -> dict:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return {}


def get_val_labels_dir(data_yaml_path: str) -> str | None:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root_path = data.get("path", "")
    val_path = data.get("val", "")
    if not val_path:
        return None
    if root_path and not os.path.isabs(val_path):
        val_path = os.path.join(root_path, val_path)
    val_path = os.path.normpath(val_path)
    parts = val_path.split(os.sep)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return os.path.normpath(os.sep.join(parts))
    parent_dir = os.path.dirname(os.path.dirname(val_path))
    val_name = os.path.basename(val_path)
    return os.path.join(parent_dir, "labels", val_name)


def count_val_label_stats(config) -> tuple:
    val_labels_dir = get_val_labels_dir(config.data_yaml)
    if not val_labels_dir or not os.path.exists(val_labels_dir):
        return {}, {}
    class_names = get_class_names_from_data_yaml(config.data_yaml)
    class_image_counts = {name: 0 for name in class_names.values()}
    class_instance_counts = {name: 0 for name in class_names.values()}
    for file_name in os.listdir(val_labels_dir):
        if not file_name.endswith(".txt"):
            continue
        file_path = os.path.join(val_labels_dir, file_name)
        appeared = set()
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) < 1:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue
            class_name = class_names.get(class_id, f"class_{class_id}")
            class_instance_counts[class_name] = class_instance_counts.get(class_name, 0) + 1
            appeared.add(class_name)
        for class_name in appeared:
            class_image_counts[class_name] = class_image_counts.get(class_name, 0) + 1
    return class_image_counts, class_instance_counts


def get_val_metrics(best_pt_path: str, config) -> object:
    from ultralytics import YOLO
    model = YOLO(best_pt_path)
    val_name = f"{config.experiment_name}_tmp_val"
    val_dir = os.path.join(config.results_dir, val_name)
    try:
        metrics = model.val(
            data=config.data_yaml, imgsz=config.imgsz, batch=config.batch,
            device=config.device, plots=False, save_txt=False, save_json=False,
            visualize=False, project=config.results_dir, name=val_name,
        )
        return metrics
    finally:
        shutil.rmtree(val_dir, ignore_errors=True)
```

- [ ] **Step 3: 编写 `linux/i18n.py`** — 从 `shared/i18n_helper.py` 复制

```python
"""i18n helpers for engine scripts."""
import json
from pathlib import Path


def load_locale(locale_dir: Path, lang: str) -> dict:
    path = locale_dir / f"{lang}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def t(loc: dict, key: str, **kwargs) -> str:
    text = loc.get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return text
```

- [ ] **Step 4: 验证模块可 import**

```bash
cd linux && python -c "from train_logger import get_timestamp; from training import build_train_kwargs; from i18n import t; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add linux/train_logger.py linux/training.py linux/i18n.py
git commit -m "feat: migrate training core, logger, i18n to linux/"
```

---

### Task 3: 编写 CLI 子命令框架 (linux/cli.py)

**Files:**
- Create: `linux/cli.py`
- Create: `linux/env_check.py`

**Interfaces:**
- Consumes: `linux/config.py` (TrainConfig, load_effective_config), `linux/training.py` (build_train_kwargs, list_experiments), `linux/i18n.py` (load_locale, t), `linux/train_logger.py` (append_train_log, append_full_val_log)
- Produces: `linux/cli.py` — `run_cli(argv: list[str]) -> None`

- [ ] **Step 1: 编写 `linux/env_check.py`** — 从 `shared/env_check.py` 复制，去掉 GUI 相关

```python
"""Pre-launch env check."""
import subprocess
import sys
import shutil

REQUIRED = [
    ("PySide6", "PySide6"),
    ("torch", "torch"),
    ("ultralytics", "ultralytics"),
    ("yaml", "pyyaml"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
]


def missing_packages():
    result = []
    for imp_name, pip_name in REQUIRED:
        try:
            __import__(imp_name)
        except ImportError:
            result.append((imp_name, pip_name))
    return result


def run_checks_and_fix():
    """Check environment; return False if user chose to exit."""
    missing = missing_packages()
    if not missing:
        return True
    print("[YOLO-LAB] 检测到缺失依赖:")
    for imp, pip in missing:
        print(f"  - {imp} (pip install {pip})")
    print(f"\n运行: pip install {' '.join(p for _, p in missing)}")
    return True
```

- [ ] **Step 2: 编写 `linux/cli.py`** — 主 CLI 框架 + train/infer/tools/config 子命令

```python
"""YOLO-LAB CLI — argparse subcommand framework."""
import os
import sys
import argparse
import json
import locale
from pathlib import Path

from config import TrainConfig, load_effective_config, save_user_config, load_user_config
from i18n import load_locale as _load_file, t as _t

_LOCALE_DIR = Path(__file__).resolve().parent / "locales"
_loc = {}


def _load_locale(lang):
    return _load_file(_LOCALE_DIR, lang)


def _detect_lang():
    try:
        system_lang, _ = locale.getdefaultlocale()
        if system_lang and system_lang[:2].lower() in ("zh", "en", "fr", "es"):
            return system_lang[:2].lower()
    except Exception:
        pass
    return "en"


# ═══════════════════════════════════════════════════════════
# Subcommand: train
# ═══════════════════════════════════════════════════════════

def _cmd_train(args):
    """Execute training."""
    from training import build_train_kwargs, list_experiments
    from train_logger import append_train_log, append_full_val_log

    cfg = load_effective_config()
    for attr in ("epochs", "imgsz", "batch", "device", "data_yaml",
                 "model_file", "results_dir", "log_dir"):
        val = getattr(args, attr, None)
        if val is not None:
            setattr(cfg, attr, val)
    if args.name:
        cfg.experiment_name = args.name

    print(_t(_loc, "train.engine.new_start", name=cfg.experiment_name))
    print(f"  model={cfg.model_file}  epochs={cfg.epochs}  imgsz={cfg.imgsz}  batch={cfg.batch}  device={cfg.device}")

    use_augment = cfg.use_augment
    if args.no_augment:
        use_augment = False

    from ultralytics import YOLO
    append_train_log(cfg, mode="cli_train", status="started", notes="CLI training")

    try:
        model = YOLO(cfg.model_file)
        train_kwargs = build_train_kwargs(cfg, use_augment)
        model.train(**train_kwargs)
        append_train_log(cfg, mode="cli_train", status="finished", notes="CLI done")
        print(_t(_loc, "msg.train_done"))
    except Exception as e:
        append_train_log(cfg, mode="cli_train", status="failed", notes=str(e))
        print(f"Training failed: {e}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════
# Subcommand: infer
# ═══════════════════════════════════════════════════════════

def _cmd_infer(args):
    """Execute inference."""
    from ultralytics import YOLO
    from paths import BEST_SEG_MODEL, PREDICT_DIR

    model_path = args.model or BEST_SEG_MODEL or "yolo11n-seg.pt"
    source = args.source
    save_dir = args.save_dir or str(Path(PREDICT_DIR) / "predict_result")
    conf = args.conf
    imgsz = args.imgsz

    if not Path(source).exists():
        print(f"Source not found: {source}")
        sys.exit(1)

    print(f"Inferencing with model={model_path}, source={source}")
    model = YOLO(model_path)
    results = model(source, save=True, project=save_dir, name="results",
                    conf=conf, imgsz=imgsz, exist_ok=True)
    print(f"Done. Results saved to {save_dir}")


# ═══════════════════════════════════════════════════════════
# Subcommand: tools
# ═══════════════════════════════════════════════════════════

def _cmd_tools(args):
    """Execute dataset tools."""
    tool_cmd = args.tool_cmd

    if tool_cmd == "split":
        _tools_split(args)
    elif tool_cmd == "labels":
        _tools_labels(args)
    elif tool_cmd == "stats":
        _tools_stats(args)
    else:
        print(f"Unknown tool: {tool_cmd}")
        sys.exit(1)


def _tools_split(args):
    source = args.source
    ratio = args.ratio
    if not Path(source).is_dir():
        print(f"Source directory not found: {source}")
        sys.exit(1)

    import shutil
    import random

    images = sorted(Path(source).glob("*"))
    images = [f for f in images if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    random.shuffle(images)
    split_idx = int(len(images) * ratio)

    train_dir = Path(source).parent / "images" / "train"
    val_dir = Path(source).parent / "images" / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for f in images[:split_idx]:
        shutil.copy2(str(f), str(train_dir / f.name))
    for f in images[split_idx:]:
        shutil.copy2(str(f), str(val_dir / f.name))

    print(f"Split: {split_idx} → train, {len(images) - split_idx} → val")


def _tools_labels(args):
    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        print(f"Directory not found: {image_dir}")
        sys.exit(1)

    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))
    for img in images:
        label_path = img.with_suffix(".txt")
        if not label_path.exists():
            label_path.touch()
    print(f"Created {len(images)} empty label files in {image_dir}")


def _tools_stats(args):
    data_yaml = args.data
    if not Path(data_yaml).is_file():
        print(f"data.yaml not found: {data_yaml}")
        sys.exit(1)

    from training import get_class_names_from_data_yaml, count_val_label_stats
    from config import TrainConfig

    cfg = TrainConfig()
    cfg.data_yaml = data_yaml

    class_names = get_class_names_from_data_yaml(data_yaml)
    img_counts, inst_counts = count_val_label_stats(cfg)

    print(f"\nDataset: {data_yaml}")
    print(f"Classes: {len(class_names)}")
    for name in class_names.values():
        print(f"  {name}: {img_counts.get(name, 0)} images, {inst_counts.get(name, 0)} instances")


# ═══════════════════════════════════════════════════════════
# Subcommand: config
# ═══════════════════════════════════════════════════════════

def _cmd_config(args):
    cfg_cmd = args.cfg_cmd
    if cfg_cmd == "init":
        _config_init()
    elif cfg_cmd == "show":
        _config_show()
    elif cfg_cmd == "set":
        _config_set(args.key, args.value)
    else:
        print("Usage: yolo-lab config {init|show|set}")


def _config_init():
    print("\n  ═══ YOLO-LAB 初始配置 ═══")
    data_yaml = input("  数据集路径 [./data.yaml]: ").strip() or "./data.yaml"
    model = input("  默认模型 [yolo11n-seg.pt]: ").strip() or "yolo11n-seg.pt"
    results = input("  输出目录 [./outputs/results]: ").strip() or "./outputs/results"
    epochs = input("  训练轮数 [150]: ").strip() or "150"
    batch = input("  批次大小 [16]: ").strip() or "16"

    cfg = TrainConfig()
    cfg.data_yaml = data_yaml
    cfg.model_file = model
    cfg.results_dir = results
    cfg.log_dir = results
    cfg.epochs = int(epochs)
    cfg.batch = int(batch)

    save_user_config(cfg)
    print(f"\n  ✓ 配置已保存到 ~/.yolo-lab/config.yaml")
    print("  运行 yolo-lab 或 yolo-lab train 开始使用。\n")


def _config_show():
    file_cfg = load_user_config()
    if not file_cfg:
        print("No config file found. Run 'yolo-lab config init' first.")
        return
    cfg = load_effective_config()
    print(f"\n  # ~/.yolo-lab/config.yaml")
    for k, v in file_cfg.items():
        print(f"  {k}: {v}")
    print()


def _config_set(set_args):
    key = set_args.key
    value = set_args.value
    file_cfg = load_user_config() or {}
    # Try to infer type
    if value.isdigit():
        value = int(value)
    elif value.lower() in ("true", "false"):
        value = value.lower() == "true"
    file_cfg[key] = value
    import yaml
    cfg_dir = Path.home() / ".yolo-lab"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(file_cfg, f, allow_unicode=True, default_flow_style=False)
    print(f"  Set {key} = {value}")


# ═══════════════════════════════════════════════════════════
# Argparse + Dispatch
# ═══════════════════════════════════════════════════════════

def run_cli(argv):
    global _loc
    parser = argparse.ArgumentParser(prog="yolo-lab", description="YOLO Segmentation Training & Inference Toolkit")
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--epochs", type=int)
    p_train.add_argument("--batch", type=int)
    p_train.add_argument("--imgsz", type=int)
    p_train.add_argument("--data", dest="data_yaml")
    p_train.add_argument("--model", dest="model_file")
    p_train.add_argument("--results", dest="results_dir")
    p_train.add_argument("--logs", dest="log_dir")
    p_train.add_argument("--device")
    p_train.add_argument("--name")
    p_train.add_argument("--no-augment", action="store_true")
    p_train.add_argument("--lang")

    # infer
    p_infer = sub.add_parser("infer", help="Run inference")
    p_infer.add_argument("--source", required=True)
    p_infer.add_argument("--model")
    p_infer.add_argument("--save-dir")
    p_infer.add_argument("--conf", type=float, default=0.25)
    p_infer.add_argument("--imgsz", type=int, default=640)

    # tools
    p_tools = sub.add_parser("tools", help="Dataset tools")
    tools_sub = p_tools.add_subparsers(dest="tool_cmd")
    p_split = tools_sub.add_parser("split", help="Split dataset")
    p_split.add_argument("--source", required=True)
    p_split.add_argument("--ratio", type=float, default=0.8)
    p_labels = tools_sub.add_parser("labels", help="Create empty labels")
    p_labels.add_argument("--image-dir", required=True)
    p_stats = tools_sub.add_parser("stats", help="Dataset statistics")
    p_stats.add_argument("--data", required=True)

    # config
    p_cfg = sub.add_parser("config", help="Manage user config")
    cfg_sub = p_cfg.add_subparsers(dest="cfg_cmd")
    cfg_sub.add_parser("init", help="Interactive config setup")
    cfg_sub.add_parser("show", help="Show current config")
    p_set = cfg_sub.add_parser("set", help="Set a config value")
    p_set.add_argument("key")
    p_set.add_argument("value")

    args = parser.parse_args(argv)

    # Load locale
    lang = getattr(args, "lang", None) or _detect_lang()
    _loc = _load_locale(lang)

    # Dispatch
    if args.command == "train":
        _cmd_train(args)
    elif args.command == "infer":
        _cmd_infer(args)
    elif args.command == "tools":
        _cmd_tools(args)
    elif args.command == "config":
        _cmd_config(args)
    else:
        parser.print_help()
```

- [ ] **Step 3: 复制 locales 到 `linux/locales/`**

```bash
cp cli/locales/*.json linux/locales/
```

- [ ] **Step 4: 验证 CLI 框架**

```bash
cd linux && python entry.py --help
```

Expected: 显示 help

```bash
cd linux && python entry.py config show
```

Expected: "No config file found"（第一次运行）或显示配置

- [ ] **Step 5: Commit**

```bash
git add linux/cli.py linux/env_check.py linux/locales/
git commit -m "feat: add CLI subcommand framework (train/infer/tools/config)"
```

---

### Task 4: 迁移 GUI 到 linux/gui/

**Files:**
- Create: `linux/gui/__init__.py`
- Create: `linux/gui/main_window.py` (adapted from `gui/main.py`)
- Modify: 所有从 `gui/` → `linux/gui/` 复制并修复 import 的文件
- Create: `linux/gui/locales/` (from `gui/locales/`)

**Interfaces:**
- Consumes: `linux/config.py`, `linux/training.py`, `linux/paths.py`, `linux/train_logger.py`, `linux/i18n.py`
- Produces: `linux/gui/main_window.py` — `run_gui()` function

**Migration mapping (all files copied + imports fixed):**

| Source | Dest |
|--------|------|
| `gui/__init__.py` | `linux/gui/__init__.py` |
| `gui/main.py` | `linux/gui/main_window.py` (with adaptations) |
| `gui/config.py` | `linux/gui/config.py` (re-export from `..config`) |
| `gui/paths.py` | → delete, use `linux/paths.py` |
| `gui/i18n.py` | `linux/gui/i18n.py` |
| `gui/styles.py` | `linux/gui/styles.py` |
| `gui/widgets.py` | `linux/gui/widgets.py` |
| `gui/workers.py` | `linux/gui/workers.py` |
| `gui/utils.py` | `linux/gui/utils.py` (fix paths import) |
| `gui/device.py` | `linux/gui/device.py` |
| `gui/gpu_manager.py` | `linux/gui/gpu_manager.py` (fix paths import) |
| `gui/model_selector.py` | `linux/gui/model_selector.py` |
| `gui/train_engine.py` | `linux/gui/train_engine.py` (fix imports) |
| `gui/infer_engine.py` | `linux/gui/infer_engine.py` (fix imports) |
| `gui/welcome_wizard.py` | `linux/gui/welcome_wizard.py` |
| `gui/tabs/*` | `linux/gui/tabs/*` |
| `gui/tools/**` | `linux/gui/tools/**` |
| `gui/locales/*` | `linux/gui/locales/*` |
| `gui/infer_task_params.json` | `linux/gui/infer_task_params.json` |
| `gui/presets.json` | `linux/gui/presets.json` |

- [ ] **Step 1: 批量复制 GUI 文件**

```bash
# 复制所有 GUI Python 文件
cp gui/__init__.py linux/gui/__init__.py
cp gui/i18n.py linux/gui/i18n.py
cp gui/styles.py linux/gui/styles.py
cp gui/widgets.py linux/gui/widgets.py
cp gui/workers.py linux/gui/workers.py
cp gui/device.py linux/gui/device.py
cp gui/gpu_manager.py linux/gui/gpu_manager.py
cp gui/model_selector.py linux/gui/model_selector.py
cp gui/train_engine.py linux/gui/train_engine.py
cp gui/infer_engine.py linux/gui/infer_engine.py
cp gui/welcome_wizard.py linux/gui/welcome_wizard.py

# 复制 tabs
cp gui/tabs/__init__.py linux/gui/tabs/__init__.py
cp gui/tabs/train_tab.py linux/gui/tabs/train_tab.py
cp gui/tabs/infer_tab.py linux/gui/tabs/infer_tab.py
cp gui/tabs/tools_tab.py linux/gui/tabs/tools_tab.py
cp gui/tabs/log_viewer_tab.py linux/gui/tabs/log_viewer_tab.py

# 复制 tools
cp -r gui/tools/dataset_tools linux/gui/tools/

# 复制资源
cp -r gui/locales linux/gui/locales
cp gui/infer_task_params.json linux/gui/
cp gui/presets.json linux/gui/ 2>/dev/null || true
```

- [ ] **Step 2: 修复 `linux/gui/utils.py`** — 修改 import 路径

```python
# 将 `from gui.paths import get_preset_file, is_frozen`
# 改为：
from paths import get_preset_file, is_frozen
```

- [ ] **Step 3: 修复 `linux/gui/config.py`**

```python
"""Re-export TrainConfig from linux package."""
from config import TrainConfig
```

- [ ] **Step 4: 修复 `linux/gui/workers.py`** — `ROOT` 路径

```python
# 原来的: ROOT = Path(__file__).resolve().parent
# 改为: ROOT = Path(__file__).resolve().parent.parent  # linux/ root
```

- [ ] **Step 5: 修复 `linux/gui/train_engine.py`** — 修改 import 路径

```python
# 原来:
#   sys.path.insert(0, str(_gui_root))
#   sys.path.insert(0, str(_gui_root.parent))  # project root for shared imports
#   from shared.config import TrainConfig
#   from shared.train_logger import append_train_log, append_full_val_log
#   from shared.train_core import ...
#   from shared.i18n_helper import load_locale as _load_file, t as _t
#   _LOCALE_DIR = Path(__file__).resolve().parent / "locales"
#
# 改为:
#   sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # linux/ root
#   from config import TrainConfig
#   from train_logger import append_train_log, append_full_val_log
#   from training import ...
#   from i18n import load_locale as _load_file, t as _t
#   _LOCALE_DIR = Path(__file__).resolve().parent / "locales"
```

- [ ] **Step 6: 修复 `linux/gui/infer_engine.py`** — 修改 import 路径

```python
# 原来:
#   sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#   from gui.paths import PREDICT_DIR, BEST_SEG_MODEL, TEST_IMAGES_DIR
#   from shared.i18n_helper import load_locale as _load_file, t as _t
#   _LOCALE_DIR = Path(__file__).resolve().parent / "locales"
#
# 改为:
#   sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # linux/ root
#   from paths import PREDICT_DIR, BEST_SEG_MODEL, TEST_IMAGES_DIR
#   from i18n import load_locale as _load_file, t as _t
#   _LOCALE_DIR = Path(__file__).resolve().parent / "locales"
```

- [ ] **Step 7: 修复所有 `gui/tabs/*.py` 中的 import** — 统一替换

`from gui.xxx` → 保持同包内相对引用或改用完整 module path

实际上同目录下的 `from gui.xxx` 保持不变即可，因为 `linux/` 在 sys.path 上，且 `linux/gui/__init__.py` 存在使 `gui` 成为可 import 的包。但需要注意 `import gui.xxx` 需要从 `linux/` 根才能 import。

更好的方案：在 `linux/gui/tabs/train_tab.py` 中，修改如下 import：

```python
# 原来: from gui.config import TrainConfig
# 改为: from config import TrainConfig
# 原来: from gui.i18n import tr, current_lang
# 改为: from gui.i18n import tr, current_lang  # 保持同包引用，因为 gui 包内的兄弟模块
```

实际上 `linux/` 添加到 sys.path 后，`gui.i18n` 可以通过 `from gui.i18n import ...` 访问（因为 gui 是个包）。但 `from config import TrainConfig` 会优先找 `linux/config.py`。

所有 `from gui.xxx import ...` 的行保持同包引用即可，但以下需要修改：

- `from gui.config import TrainConfig` → `from config import TrainConfig`（linux/config.py）
- `from gui.paths import ...` → `from paths import ...`（linux/paths.py）
- `from gui.utils import engine_cmd, log_append, model_file_ok, ...` → 保持 `from gui.utils import ...` 或改为 `from gui.utils import ...`（同包）

实际上最简单的做法：在 `entry.py` 和各个文件里确保 `linux/` 在 sys.path 上，然后：
- `from config import TrainConfig` → 找到 `linux/config.py` ✓
- `from training import ...` → 找到 `linux/training.py` ✓
- `from gui.i18n import ...` → 找到 `linux/gui/i18n.py` ✓ (因为 gui 是包)
- `from gui.workers import ...` → 找到 `linux/gui/workers.py` ✓

所以只需修改跨目录的 import：
- `from shared.config import ...` → `from config import ...`
- `from shared.train_core import ...` → `from training import ...`
- `from shared.train_logger import ...` → `from train_logger import ...`
- `from shared.i18n_helper import ...` → `from i18n import ...`
- `from shared.env_check import ...` → `from env_check import ...`
- `from gui.paths import ...` → `from paths import ...` (or keep `from paths import ...` since paths is at linux/ root)

Let me handle this in the plan. I'll need to write a sed command to batch-replace imports across all the copied files.

- [ ] **Step 8: 批量修复 import 路径**

```bash
cd linux

# shared.xxx → root-level modules
find gui/ -name "*.py" -exec sed -i \
  -e 's/from shared\.config import/from config import/g' \
  -e 's/from shared\.train_core import/from training import/g' \
  -e 's/from shared\.train_logger import/from train_logger import/g' \
  -e 's/from shared\.i18n_helper import/from i18n import/g' \
  -e 's/from shared\.env_check import/from env_check import/g' \
  -e 's/from shared import/from /g' \
  {} +

# gui.paths → paths (now at linux/ root)
find gui/ -name "*.py" -exec sed -i \
  -e 's/from gui\.paths import/from paths import/g' \
  -e 's/from gui\.config import/from config import/g' \
  {} +
```

- [ ] **Step 9: 创建 `linux/gui/main_window.py`** — 从 `gui/main.py` 改编

```python
"""
YOLO 分割训练 / 推理桌面界面 — Apple 风格简约设计
启动：python linux/entry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

LINUX_ROOT = Path(__file__).resolve().parent.parent
if str(LINUX_ROOT) not in sys.path:
    sys.path.insert(0, str(LINUX_ROOT))

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

        self._train_tab = TrainTab()
        self._infer_tab = InferTab()
        self._tools_tab = ToolsTab()
        self._log_viewer_tab = LogViewerTab()

        self._train_tab.closing.connect(self._on_worker_closing)
        self._infer_tab.closing.connect(self._on_worker_closing)

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

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._tabs)

        QShortcut(QKeySequence("Ctrl+Return"), self, self._on_ctrl_enter)

    def _toggle_dark_mode(self):
        self._dark_mode = not self._dark_mode
        self._dark_btn.setText("🌙" if self._dark_mode else "☀")
        apply_theme_to_widgets(self, self._dark_mode)

    def _on_lang_changed(self, idx):
        lang = self._lang_combo.itemData(idx)
        if lang:
            set_language(lang)
            apply_language(self)

    def _on_ctrl_enter(self):
        idx = self._tabs.currentIndex()
        if idx == 0:
            self._train_tab.on_ctrl_enter()
        elif idx == 1:
            self._infer_tab._on_start_infer()

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


def _env_check():
    from env_check import run_checks_and_fix
    if not run_checks_and_fix():
        sys.exit(0)


def run_gui():
    _env_check()
    app = QApplication(sys.argv)
    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    font.setPixelSize(FONT_SIZE)
    app.setFont(font)
    w = MainWindow()
    w.show()

    from gui.welcome_wizard import is_first_run, WelcomeWizard
    if is_first_run():
        wizard = WelcomeWizard(w)
        wizard.exec()

    sys.exit(app.exec())


def _run_engine_mode():
    """Entry point when launched as --engine-{train,infer,tool} by a subprocess."""
    import traceback

    from gui.gpu_manager import get_gpu_dir, has_gpu_ready_marker
    if has_gpu_ready_marker():
        gpu_dir = str(get_gpu_dir())
        if gpu_dir not in sys.path:
            sys.path.insert(0, gpu_dir)

    from gui import train_engine, infer_engine

    mode = sys.argv[1]
    sys.argv.pop(1)

    try:
        if mode == "--engine-train":
            args = train_engine.parse_args()
            train_engine.run_non_interactive(args)
        elif mode == "--engine-infer":
            import argparse
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
                "gui.tools.dataset_tools.create_empty_labels",
                "gui.tools.dataset_tools.split_train_val.split_random_with_labels",
                "gui.tools.dataset_tools.split_train_val_test.split_random_with_labels",
                "gui.tools.dataset_tools.split_train_val.split_every_5th_with_labels",
                "gui.tools.dataset_tools.split_images_only.split_random_images_only",
                "gui.tools.dataset_tools.split_images_only.split_every_5th_images_only",
            ]
            mod = importlib.import_module(TOOL_MODULES[tool_idx])
            mod.run(**tool_args)
    except SystemExit:
        raise
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    _env_check()
    if len(sys.argv) > 1 and sys.argv[1].startswith("--engine-"):
        _run_engine_mode()
    else:
        run_gui()
```

- [ ] **Step 10: 修复 `linux/gui/workers.py` 中的 ROOT**

```python
# 将 ROOT = Path(__file__).resolve().parent
# 改为：ROOT = Path(__file__).resolve().parent.parent  # linux/ directory
```

- [ ] **Step 11: 验证 GUI import**

```bash
cd linux && python -c "from gui.i18n import tr; print('GUI import OK')"
```

Expected: `GUI import OK` (may need DISPLAY variable for Qt, but import alone should work)

- [ ] **Step 12: Commit**

```bash
git add linux/gui/
git commit -m "feat: migrate GUI to linux/gui/ with fixed imports"
```

---

### Task 5: AppImage 打包 + install.sh + desktop

**Files:**
- Create: `linux/build.sh`
- Create: `linux/install.sh`
- Create: `linux/yolo_lab.desktop`
- Create: `linux/yolo_lab.spec` (PyInstaller spec)

**Interfaces:**
- N/A (standalone scripts)

- [ ] **Step 1: 编写 `linux/yolo_lab.spec`**

```python
# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

LINUX_ROOT = Path(SPECPATH).resolve().parent

_binaries = []
if sys.platform == "linux":
    _expat_path = Path(sys.prefix) / "lib" / "libexpat.so.1"
    if _expat_path.is_file():
        _binaries.append((str(_expat_path), "."))

a = Analysis(
    [str(LINUX_ROOT / "entry.py")],
    pathex=[str(LINUX_ROOT), str(LINUX_ROOT / "gui"), str(LINUX_ROOT / "gui" / "tools")],
    binaries=_binaries,
    datas=[
        (str(LINUX_ROOT / "gui" / "locales"), "locales"),
        (str(LINUX_ROOT / "gui" / "infer_task_params.json"), "gui"),
        (str(LINUX_ROOT / "locales"), "locales"),
    ],
    hiddenimports=[
        "config",
        "training",
        "train_logger",
        "i18n",
        "env_check",
        "cli",
        "ultralytics",
        "ultralytics.nn",
        "ultralytics.nn.modules",
        "ultralytics.nn.tasks",
        "ultralytics.data",
        "ultralytics.data.build",
        "ultralytics.utils",
        "ultralytics.utils.checks",
        "ultralytics.utils.downloads",
        "ultralytics.engine",
        "ultralytics.engine.model",
        "ultralytics.engine.results",
        "torch",
        "cv2",
        "numpy",
        "yaml",
        "matplotlib",
        "matplotlib.backends.backend_agg",
        "PIL",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "gui.tabs",
        "gui.tabs.train_tab",
        "gui.tabs.infer_tab",
        "gui.tabs.tools_tab",
        "gui.tabs.log_viewer_tab",
        "gui.utils",
        "gui.tools.dataset_tools.create_empty_labels",
        "gui.tools.dataset_tools.split_train_val.split_random_with_labels",
        "gui.tools.dataset_tools.split_train_val_test.split_random_with_labels",
        "gui.tools.dataset_tools.split_train_val.split_every_5th_with_labels",
        "gui.tools.dataset_tools.split_images_only.split_random_images_only",
        "gui.tools.dataset_tools.split_images_only.split_every_5th_images_only",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torchaudio",
        "nvidia",
        "nvidia.cublas",
        "nvidia.cudnn",
        "nvidia.cufft",
        "nvidia.curand",
        "nvidia.cusolver",
        "nvidia.cusparse",
        "nvidia.nccl",
        "nvidia.nvtx",
        "nvidia.cuda_runtime",
        "triton",
        "triton.language",
        "triton.runtime",
        "PySide6.QtWebEngine",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebChannel",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtBluetooth",
        "PySide6.QtNfc",
        "PySide6.QtSensors",
        "PySide6.QtSerialPort",
        "PySide6.QtPositioning",
        "PySide6.QtLocation",
        "PySide6.QtRemoteObjects",
        "PySide6.QtTextToSpeech",
        "polars",
        "polars_runtime_32",
        "torch.test",
        "torch.testing",
        "torch.distributed",
        "scipy",
        "pandas",
        "tcl",
        "tkinter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="YoloLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(LINUX_ROOT / "assets" / "icon.png"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="YoloLab",
)
```

- [ ] **Step 2: 编写 `linux/build.sh`**

```bash
#!/bin/bash
set -euo pipefail

VERSION="${1:-0.2.0}"
LINUX_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="$LINUX_DIR/dist"
SPEC="$LINUX_DIR/yolo_lab.spec"

echo "==> Building YoloLab v${VERSION} for Linux..."

# Clean
rm -rf "$DIST_DIR" "$LINUX_DIR/build"

# PyInstaller
pyinstaller --distpath "$DIST_DIR" --workpath "$LINUX_DIR/build" "$SPEC"

# Rename output to versioned AppImage name
# Note: actual AppImage packaging may need additional tooling;
# this produces the COLLECT directory that can be packaged.
echo "==> Build complete: $DIST_DIR/YoloLab"
```

- [ ] **Step 3: 编写 `linux/install.sh`**

```bash
#!/bin/bash
set -euo pipefail

INSTALL_DIR="/opt/YoloLab"
BIN_LINK="/usr/local/bin/yolo-lab"
ICON_DIR="/usr/share/icons/hicolor/256x256/apps"
APP_DIR="/usr/share/applications"

if [ "$(id -u)" -ne 0 ]; then
    echo "请使用 sudo 运行: sudo ./install.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APPIMAGE=$(ls "$SCRIPT_DIR/dist"/YoloLab*.AppImage 2>/dev/null | head -1)

if [ -z "$APPIMAGE" ]; then
    echo "未找到 AppImage，请先运行 build.sh"
    exit 1
fi

echo "==> 安装 YOLO-LAB..."

# Install app
mkdir -p "$INSTALL_DIR"
cp "$APPIMAGE" "$INSTALL_DIR/YoloLab.AppImage"
chmod +x "$INSTALL_DIR/YoloLab.AppImage"

# Install icon
mkdir -p "$ICON_DIR"
cp "$SCRIPT_DIR/assets/icon.png" "$ICON_DIR/yolo-lab.png"

# Create symlink
ln -sf "$INSTALL_DIR/YoloLab.AppImage" "$BIN_LINK"

# Install desktop entry
mkdir -p "$APP_DIR"
cp "$SCRIPT_DIR/yolo_lab.desktop" "$APP_DIR/"

echo "✓ 安装完成！"
echo "  GUI: 从应用菜单启动 或 终端输入 yolo-lab"
echo "  CLI: yolo-lab train --epochs 100"
```

- [ ] **Step 4: 编写 `linux/yolo_lab.desktop`**

```ini
[Desktop Entry]
Name=YOLO-LAB
Comment=YOLO Segmentation Training & Inference Toolkit
Exec=/opt/YoloLab/YoloLab.AppImage
Icon=yolo-lab
Type=Application
Categories=Science;ArtificialIntelligence;
Terminal=false
```

- [ ] **Step 5: 验证打包脚本语法**

```bash
bash -n linux/build.sh && echo "build.sh OK"
bash -n linux/install.sh && echo "install.sh OK"
```

- [ ] **Step 6: Commit**

```bash
git add linux/build.sh linux/install.sh linux/yolo_lab.desktop linux/yolo_lab.spec
git commit -m "feat: add AppImage build, install script, and desktop entry for Linux"
```

---

### Task 6: 清理旧文件 + 更新 CI + 创建占位目录

**Files:**
- Delete: `shared/`, `gui/`, `cli/`, `packaging/` (except references)
- Create: `windows/`, `macos/` (empty shells)
- Modify: `.github/workflows/release.yml` (if exists)

**Interfaces:**
- N/A (cleanup)

- [ ] **Step 1: 确认 linux/ 功能独立可用**

```bash
# Test CLI
cd linux && python entry.py config show
# Test GUI import
cd linux && python -c "from gui.main_window import run_gui; print('OK')"
```

- [ ] **Step 2: 删除旧文件**

```bash
git rm -r shared/
git rm -r gui/
git rm -r cli/
git rm -r packaging/
```

- [ ] **Step 3: 创建 windows/ 和 macos/ 占位**

```bash
mkdir -p windows/assets macos/assets
```

- [ ] **Step 4: 编写 `windows/README.md`**

```markdown
# Windows — 预留

Windows 版本的 YOLO-LAB 将在此目录开发。

计划：PySide6 GUI + NSIS 安装包 + CLI 入口。
```

- [ ] **Step 5: 编写 `macos/README.md`**

```markdown
# macOS — 预留

macOS 版本的 YOLO-LAB 将在此目录开发。

计划：PySide6 GUI + DMG 打包 + CLI 入口。
```

- [ ] **Step 6: 更新 `.github/workflows/`** — 如果有 CI 文件，修改构建路径

检查是否存在：

```bash
ls .github/workflows/ 2>/dev/null || echo "No CI workflows"
```

如果存在，暂时注释或修改为使用 `linux/` 路径。

- [ ] **Step 7: 更新 `tests/`** — 暂不删除，但需注释原有导入（后续适配）

保持 tests/ 不动，让后续 Task 处理。

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: remove old shared/gui/cli dirs, create windows/macos placeholders"
```

---

### Task 7: 重写 README

**Files:**
- Modify: `README.md`

**Interfaces:**
- N/A (documentation)

- [ ] **Step 1: 编写新 README**

新 README 内容 — 反映按系统分文件夹的新架构，Linux 为主：

```markdown
# YOLO-LAB

YOLO 模型训练与推理工具集 — 一个应用，两种用法：双击启动 GUI，终端使用 CLI。

## 目录结构

| 目录 | 说明 | 状态 |
|------|------|------|
| `linux/`   | Linux 版应用 | ✅ 可用 |
| `windows/` | Windows 版应用 | 🔜 预留 |
| `macos/`   | macOS 版应用 | 🔜 预留 |

每个系统文件夹是完全独立的应用（代码、图标、打包脚本全在里面）。

---

## 快速开始 (Linux)

### 从源码运行

```bash
git clone https://github.com/Liujingze11/YOLO-LAB.git
cd YOLO-LAB/linux

# 安装依赖
pip install -r requirements.txt

# 启动 GUI
python entry.py

# 使用 CLI
python entry.py train --epochs 200 --batch 8
python entry.py infer --source ./image.jpg --model ./best.pt
python entry.py tools split --source ./all_images --ratio 0.8
python entry.py config init
```

### 安装包

前往 [Releases](https://github.com/Liujingze11/YOLO-LAB/releases) 下载 `YoloLab.AppImage`：

```bash
chmod +x YoloLab-*.AppImage
sudo ./linux/install.sh    # 一键安装

# 安装后：
yolo-lab                    # 启动 GUI
yolo-lab train --epochs 100 # CLI 训练
yolo-lab config init        # 首次配置向导
```

---

## CLI 命令

```
yolo-lab
├── train      训练模型
│   --epochs, --batch, --imgsz, --data, --model, --device, --name, --no-augment
├── infer      推理预测
│   --source, --model, --save-dir, --conf, --imgsz
├── tools      数据集工具
│   ├── split   --source, --ratio
│   ├── labels  --image-dir
│   └── stats   --data
└── config     配置管理
    ├── init    交互式创建配置
    ├── show    显示当前配置
    └── set     key value 修改单个配置项
```

## 配置文件

CLI 参数 > `~/.yolo-lab/config.yaml` > 代码默认值。

```bash
yolo-lab config init   # 交互式创建，配置一次永久生效
yolo-lab config show   # 查看当前配置
```

---

## GPU 加速

进入 GUI 训练标签页，Device 下拉选择 GPU 即可自动检测下载 CUDA 组件。

---

## 本地构建

```bash
cd linux/
./build.sh --version 0.2.0
# 输出：dist/YoloLab/
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for unified GUI/CLI architecture"
```

---

### Task 8: 最终验证

- [ ] **Step 1: 验证 CLI 全链路**

```bash
cd linux && python entry.py --help
cd linux && python entry.py config show
cd linux && python entry.py train --help
cd linux && python entry.py infer --help
cd linux && python entry.py tools --help
cd linux && python entry.py tools split --help
```

- [ ] **Step 2: 验证 GUI 可导入**

```bash
cd linux && python -c "
from gui.main_window import run_gui, MainWindow
from gui.tabs.train_tab import TrainTab
from gui.tabs.infer_tab import InferTab
from gui.tabs.tools_tab import ToolsTab
from gui.i18n import tr
from config import TrainConfig, load_effective_config
from training import build_train_kwargs
from cli import run_cli
print('All imports OK')
"
```

- [ ] **Step 3: 检查无 broken import**

```bash
grep -r "from shared\." linux/ && echo "ERROR: stale shared imports found" || echo "OK: no stale shared imports"
grep -r "from gui\.paths" linux/gui/ && echo "ERROR: stale gui.paths imports found" || echo "OK: no stale gui.paths imports"
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "chore: final verification — all imports clean"
```
