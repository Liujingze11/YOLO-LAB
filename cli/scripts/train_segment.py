import os
os.environ["MPLBACKEND"] = "Agg"

import json
import locale
import yaml
import shutil
import argparse
from pathlib import Path
from ultralytics import YOLO
from config import TrainConfig
from train_logger import append_train_log, append_full_val_log

# ── i18n ──────────────────────────────────────────────────

LOCALE_DIR = Path(__file__).resolve().parent.parent / "locales"

def _detect_lang():
    try:
        system_lang, _ = locale.getdefaultlocale()
        if system_lang:
            code = system_lang[:2].lower()
            if code in ("zh", "en", "fr", "es"):
                return code
    except Exception:
        pass
    return "en"

def _load_locale(lang):
    path = LOCALE_DIR / f"{lang}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _t(loc, key, **kwargs):
    text = loc.get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return text

# ── 配置 ──────────────────────────────────────────────────

CONFIG = TrainConfig()


# ── 工具函数 ──────────────────────────────────────────────

def ask_confirm_train(mode, pt_path, config):
    print(f"\n------------------------------")
    print(_t(_loc, "confirm.title", mode=mode))
    print(_t(_loc, "confirm.pt_file", path=pt_path))
    print(_t(_loc, "confirm.data_yaml", path=config.data_yaml))
    print(_t(_loc, "confirm.exp_name", name=config.experiment_name))
    print(_t(_loc, "confirm.epochs", epochs=config.epochs))
    print("------------------------------")

    confirm = input(_t(_loc, "confirm.prompt")).strip().lower()
    if confirm != "y":
        print(f"\n{_t(_loc, 'confirm.cancelled')}")
        return False
    return True


def list_experiments(results_dir):
    if not os.path.exists(results_dir):
        print(f"\n{_t(_loc, 'results.not_found', dir=results_dir)}")
        return []

    folders = []
    for name in os.listdir(results_dir):
        full_path = os.path.join(results_dir, name)
        if os.path.isdir(full_path):
            folders.append(name)
    folders.sort()
    return folders


# ── 命令行参数 ────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO training script")
    parser.add_argument("--epochs", type=int, default=None, help="training epochs")
    parser.add_argument("--imgsz", type=int, default=None, help="input image size")
    parser.add_argument("--batch", type=int, default=None, help="batch size")
    parser.add_argument("--device", type=str, default=None, help="device: 0 / 0,1 / cpu")
    parser.add_argument("--name", type=str, default=None, help="experiment name")
    parser.add_argument("--lang", type=str, default=None, help="language: zh/en/fr/es (auto-detect if not set)")
    return parser.parse_args()


def override_config_from_args(config, args):
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.imgsz is not None:
        config.imgsz = args.imgsz
    if args.batch is not None:
        config.batch = args.batch
    if args.device is not None:
        config.device = args.device
    if args.name is not None:
        config.experiment_name = args.name
    return config


# ── 数据增强 ──────────────────────────────────────────────

def ask_use_augment(config):
    status = _t(_loc, "augment.status_on") if config.use_augment else _t(_loc, "augment.status_off")
    print(f"\n------------------------------")
    print(_t(_loc, "augment.title"))
    print(_t(_loc, "augment.current", status=status))
    print("------------------------------")

    choice = input(_t(_loc, "augment.prompt")).strip().lower()
    if choice == "y":
        return True
    elif choice == "n":
        return False
    else:
        return config.use_augment


def build_train_kwargs(config, use_augment):
    kwargs = {
        "data": config.data_yaml,
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "device": config.device,
        "project": config.results_dir,
        "name": config.experiment_name,
        "plots": False,
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


# ── 数据集与验证 ──────────────────────────────────────────

def get_class_names_from_data_yaml(data_yaml_path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    else:
        return {}


def get_val_labels_dir(data_yaml_path):
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


def count_val_label_stats(config):
    val_labels_dir = get_val_labels_dir(config.data_yaml)
    if not val_labels_dir or not os.path.exists(val_labels_dir):
        print(f"\n{_t(_loc, 'val.no_labels_dir', dir=val_labels_dir)}")
        return {}, {}

    class_names = get_class_names_from_data_yaml(config.data_yaml)
    class_image_counts = {name: 0 for name in class_names.values()}
    class_instance_counts = {name: 0 for name in class_names.values()}

    for file_name in os.listdir(val_labels_dir):
        if not file_name.endswith(".txt"):
            continue
        file_path = os.path.join(val_labels_dir, file_name)
        appeared_in_this_image = set()
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
            appeared_in_this_image.add(class_name)
        for class_name in appeared_in_this_image:
            class_image_counts[class_name] = class_image_counts.get(class_name, 0) + 1

    return class_image_counts, class_instance_counts


def get_val_metrics(best_pt_path, config):
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


def log_validation_result(config, mode, notes=""):
    if not os.path.exists(config.best_pt):
        print(f"\n{_t(_loc, 'val.no_best_pt', path=config.best_pt)}")
        return
    try:
        metrics = get_val_metrics(config.best_pt, config)
        class_image_counts, class_instance_counts = count_val_label_stats(config)
        append_full_val_log(
            config=config, mode=mode, metrics=metrics,
            class_image_counts=class_image_counts,
            class_instance_counts=class_instance_counts, notes=notes,
        )
        print(f"\n{_t(_loc, 'val.logged')}")
    except Exception as e:
        print(f"\n{_t(_loc, 'val.failed', err=e)}")


# ── 训练流程 ──────────────────────────────────────────────

def start_new_training(config):
    mode_label = _t(_loc, "train.new_mode_label")
    if not ask_confirm_train(mode_label, config.model_file, config):
        return
    use_augment = ask_use_augment(config)
    aug_label = _t(_loc, "augment.status_on") if use_augment else _t(_loc, "augment.status_off")
    append_train_log(config, mode="new_train", status="started",
                     notes=_t(_loc, "log.new_started", aug=aug_label))

    try:
        model = YOLO(config.model_file)
        train_kwargs = build_train_kwargs(config, use_augment)
        model.train(**train_kwargs)
        append_train_log(config, mode="new_train", status="finished",
                         notes=_t(_loc, "log.new_finished", aug=aug_label))
        log_validation_result(config, mode="new_train", notes=_t(_loc, "log.new_val"))
    except Exception as e:
        if os.path.exists(config.best_pt):
            append_train_log(config, mode="new_train", status="finished",
                             notes=_t(_loc, "log.val_retry", err=e))
            print(f"\n{_t(_loc, 'train.completed_but_val_failed', err=e)}")
            log_validation_result(config, mode="new_train",
                                  notes=_t(_loc, "log.val_retry", err=e))
        else:
            append_train_log(config, mode="new_train", status="failed",
                             notes=_t(_loc, "log.failed", err=e))
            print(f"\n{_t(_loc, 'train.failed', err=e)}")


def resume_training(config):
    if not os.path.exists(config.last_pt):
        print(f"\n{_t(_loc, 'resume.not_found', path=config.last_pt)}")
        choice = input(_t(_loc, "resume.fallback_prompt")).strip().lower()
        if choice == "y":
            start_new_training(config)
        else:
            print(_t(_loc, "resume.cancelled"))
        return

    mode_label = _t(_loc, "train.resume_mode_label")
    if not ask_confirm_train(mode_label, config.last_pt, config):
        return

    append_train_log(config, mode="resume_train", status="started",
                     notes=_t(_loc, "log.resume_started"))
    try:
        model = YOLO(config.last_pt)
        model.train(resume=True)
        append_train_log(config, mode="resume_train", status="finished",
                         notes=_t(_loc, "log.resume_finished"))
        log_validation_result(config, mode="resume_train", notes=_t(_loc, "log.resume_val"))
    except Exception as e:
        append_train_log(config, mode="resume_train", status="failed",
                         notes=_t(_loc, "log.failed", err=e))
        print(f"\n{_t(_loc, 'resume.failed', err=e)}")


def train_from_previous_best(config):
    folders = list_experiments(config.results_dir)
    if not folders:
        print(f"\n{_t(_loc, 'history.empty')}")
        return

    print(f"\n{_t(_loc, 'history.list_title')}")
    for i, folder in enumerate(folders, 1):
        print(f"{i} - {folder}")

    choice = input(_t(_loc, "history.select_prompt")).strip()
    if not choice.isdigit():
        print(_t(_loc, "history.invalid_input"))
        return

    idx = int(choice) - 1
    if idx < 0 or idx >= len(folders):
        print(_t(_loc, "history.out_of_range"))
        return

    selected_exp = folders[idx]
    selected_best_pt = os.path.join(config.results_dir, selected_exp, "weights", "best.pt")
    if not os.path.exists(selected_best_pt):
        print(f"\n{_t(_loc, 'history.no_best_pt', path=selected_best_pt)}")
        return

    print(f"\n{_t(_loc, 'history.selected', name=selected_exp)}")

    mode_label = _t(_loc, "train.finetune_mode_label")
    if not ask_confirm_train(mode_label, selected_best_pt, config):
        return

    use_augment = ask_use_augment(config)
    aug_label = _t(_loc, "augment.status_on") if use_augment else _t(_loc, "augment.status_off")

    append_train_log(config, mode="train_from_best", status="started",
                     notes=_t(_loc, "log.finetune_started", exp=selected_exp, aug=aug_label))
    try:
        model = YOLO(selected_best_pt)
        train_kwargs = build_train_kwargs(config, use_augment)
        model.train(**train_kwargs)
        append_train_log(config, mode="train_from_best", status="finished",
                         notes=_t(_loc, "log.finetune_finished", exp=selected_exp, aug=aug_label))
        log_validation_result(config, mode="train_from_best",
                              notes=_t(_loc, "log.finetune_val", exp=selected_exp))
    except Exception as e:
        if os.path.exists(config.best_pt):
            append_train_log(config, mode="train_from_best", status="finished",
                             notes=_t(_loc, "log.val_retry", err=e))
            print(f"\n{_t(_loc, 'train.completed_but_val_failed', err=e)}")
            log_validation_result(config, mode="train_from_best",
                                  notes=_t(_loc, "log.val_retry", err=e))
        else:
            append_train_log(config, mode="train_from_best", status="failed",
                             notes=_t(_loc, "log.failed", err=e))
            print(f"\n{_t(_loc, 'train.failed', err=e)}")


# ── 主入口 ────────────────────────────────────────────────

_loc = {}

def main():
    global CONFIG, _loc
    args = parse_args()
    lang = args.lang or _detect_lang()
    _loc = _load_locale(lang)
    CONFIG = override_config_from_args(CONFIG, args)

    print(_t(_loc, "mode.select"))
    print(_t(_loc, "mode.1"))
    print(_t(_loc, "mode.2"))
    print(_t(_loc, "mode.3"))
    choice = input(_t(_loc, "mode.prompt") + "\n").strip()

    if choice == "1":
        start_new_training(CONFIG)
    elif choice == "2":
        resume_training(CONFIG)
    elif choice == "3":
        train_from_previous_best(CONFIG)
    else:
        print(_t(_loc, "mode.invalid"))


if __name__ == "__main__":
    main()
