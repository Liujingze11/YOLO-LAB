# YOLO Lab CLI

[中文](README_zh.md) | [Français](README_fr.md) | [Español](README_es.md)

Command-line YOLO segmentation training tool built on Ultralytics.

## Features

- Three training modes: New / Resume / Fine-tune
- Toggleable data augmentation
- Automatic validation with CSV logging (overall + per-class metrics)
- Experiment isolation: each run creates independent result directories and logs
- CLI parameter overrides (`--epochs`, `--imgsz`, `--batch`, `--device`, `--name`)

## Quick Start

```bash
git clone https://github.com/Liujingze11/YOLO-LAB-CLI.git
cd YOLO-LAB-CLI
pip install -r requirements.txt
python scripts/train_segment.py
```

## Requirements

- Python 3.8+
- ultralytics, PyYAML

```bash
pip install ultralytics pyyaml
```

## Project Structure

```
YOLO-LAB-CLI/
├── scripts/                # Core training scripts
│   ├── train_segment.py    # Main training script (interactive)
│   ├── config.py           # TrainConfig data class
│   ├── paths.py            # Path definitions
│   ├── train_logger.py     # CSV logging
│   └── predict_test.py     # Inference testing
├── dataset_tools/          # Dataset splitting & label utilities
│   ├── create_empty_labels.py
│   ├── split_train_val/
│   ├── split_train_val_test/
│   └── split_images_only/
├── pretrained_models/      # Pretrained models
├── data.yaml               # Dataset configuration
└── requirements.txt
```

## Training Modes

Run `python scripts/train_segment.py` and choose:

- **1** — New training from initial weights
- **2** — Resume from last.pt
- **3** — Fine-tune from historical best.pt

## CLI Options

```bash
python scripts/train_segment.py --epochs 200 --imgsz 1280 --batch 8 --device 0 --name my_experiment
```

## data.yaml Format

```yaml
path: ./data/datasets
train: images/train
val: images/val
names:
  0: class_a
  1: class_b
```

## Outputs

- Results: `result/<experiment_name>/weights/` (best.pt, last.pt)
- CSV logs: `train_logs/`

## License

MIT
