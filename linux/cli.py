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
    from train_logger import append_train_log

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

    print(f"Split: {split_idx} -> train, {len(images) - split_idx} -> val")


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
        _config_set(args)
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
