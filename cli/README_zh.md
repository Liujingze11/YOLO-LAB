# YOLO Lab CLI

[English](README.md) | [Français](README_fr.md) | [Español](README_es.md)

YOLO 分割模型命令行训练工具，基于 Ultralytics。

## 功能

- 三种训练模式：新训练 / 续训 / 微调
- 数据增强开关控制
- 自动验证并记录 CSV 日志（整体 + 每类指标）
- 实验隔离：每次训练生成独立的结果目录和日志
- 命令行参数覆盖配置（`--epochs`, `--imgsz`, `--batch`, `--device`, `--name`）
- 自动检测系统语言（zh/en/fr/es），支持 `--lang` 手动指定

## 快速开始

```bash
git clone https://github.com/Liujingze11/YOLO-LAB-CLI.git
cd YOLO-LAB-CLI
pip install -r requirements.txt
python scripts/train_segment.py
```

## 依赖

- Python 3.8+
- ultralytics, PyYAML

```bash
pip install ultralytics pyyaml
```

## 项目结构

```
YOLO-LAB-CLI/
├── scripts/                # 训练核心脚本
│   ├── train_segment.py    # 训练主脚本（交互式）
│   ├── config.py           # TrainConfig 配置类
│   ├── paths.py            # 路径定义
│   ├── train_logger.py     # CSV 日志
│   └── predict_test.py     # 推理测试
├── dataset_tools/          # 数据集分割 & 标签工具
│   ├── create_empty_labels.py
│   ├── split_train_val/
│   ├── split_train_val_test/
│   └── split_images_only/
├── pretrained_models/      # 预训练模型
├── data.yaml               # 数据集配置
└── requirements.txt
```

## 训练模式

运行 `python scripts/train_segment.py` 后选择：

- **1** — 新训练，从初始权重开始
- **2** — 续训，从上次 `last.pt` 继续
- **3** — 微调，基于历史实验 `best.pt`

## 命令行参数

```bash
python scripts/train_segment.py --epochs 200 --imgsz 1280 --batch 8 --device 0 --name my_experiment
```

语言默认根据系统自动检测，也可通过 `--lang` 指定：

```bash
python scripts/train_segment.py --lang zh   # 中文
python scripts/train_segment.py --lang en   # English
python scripts/train_segment.py --lang fr   # Français
python scripts/train_segment.py --lang es   # Español
```

## data.yaml 格式

```yaml
path: ./data/datasets
train: images/train
val: images/val
names:
  0: class_a
  1: class_b
```

## 输出

- 实验结果：`result/<experiment_name>/weights/` (best.pt, last.pt)
- CSV 日志：`train_logs/`

## License

MIT
