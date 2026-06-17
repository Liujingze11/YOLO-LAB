# YOLO-LAB 统一 GUI/CLI 设计规格

> 状态：草稿 | 日期：2026-06-17

## 1. 概述

### 目标

用户下载 YOLO-LAB 后，既能双击图标启动图形界面（GUI），也能在终端通过命令行（CLI）使用全部功能。一个仓库，一个安装包，两种用法。

### 动机

- 当前 `shared/`、`gui/`、`cli/` 是拆分状态，且与外部 `YOLO-LAB-CLI`、`YOLO-LAB-GUI` 仓库耦合
- 用户需要灵活的使用方式：日常用 GUI，批量/脚本用 CLI
- 不同系统（Linux/macOS/Windows）的 GUI 可能有差异，需独立演进

### 核心原则

- **每系统独立**：`linux/`、`windows/`、`macos/` 各自是完全自包含的应用（代码、配置、图标、打包脚本）
- **统一入口**：同一可执行文件，无参启动 GUI，有子命令走 CLI
- **配置文件驱动**：路径等长参数存在 YAML 配置文件中，CLI 参数仅用于覆盖
- **先做 Linux**：windows/ 和 macos/ 留空壳占位，后续再填

---

## 2. 目录结构

```
YOLO-LAB/
├── linux/                     # 🔥 当前开发（完全独立）
│   ├── gui/                   # PySide6 GUI
│   │   ├── __init__.py
│   │   ├── main_window.py     # 主窗口 + 入口启动
│   │   ├── tabs/
│   │   │   ├── __init__.py
│   │   │   ├── train_tab.py
│   │   │   ├── infer_tab.py
│   │   │   ├── tools_tab.py
│   │   │   └── log_viewer_tab.py
│   │   ├── widgets.py         # 通用控件工厂
│   │   ├── styles.py          # 主题/样式
│   │   ├── workers.py         # QThread 后台线程
│   │   ├── train_engine.py    # 训练子进程引擎
│   │   ├── infer_engine.py    # 推理子进程引擎
│   │   ├── model_selector.py  # 模型选择/下载
│   │   ├── device.py          # GPU 检测
│   │   ├── gpu_manager.py     # GPU 组件下载
│   │   ├── welcome_wizard.py  # 首次使用引导
│   │   ├── i18n.py            # 国际化
│   │   └── utils.py           # 工具函数
│   │
│   ├── cli.py                 # CLI argparse 子命令框架
│   ├── training.py            # 训练/验证核心逻辑
│   ├── config.py              # TrainConfig 数据类 + 用户配置管理
│   ├── i18n.py                # CLI 端国际化
│   ├── paths.py               # 路径解析工具
│   ├── train_logger.py        # 训练日志
│   │
│   ├── data.yaml              # 默认数据集配置模板
│   ├── requirements.txt       # Linux 依赖
│   │
│   ├── assets/                # Linux 专用资源
│   │   ├── icon.png
│   │   └── icon.svg
│   │
│   ├── entry.py               # 统一入口：无参→GUI，有参→CLI
│   ├── build.sh               # AppImage 构建脚本
│   ├── install.sh             # 系统安装（复制+符号链接+desktop文件）
│   └── yolo_lab.desktop       # 桌面入口文件
│
├── windows/                   # 预留，暂不开发
│   ├── gui/
│   ├── assets/
│   ├── cli.py
│   ├── entry.py
│   ├── build.bat
│   └── ...
│
├── macos/                     # 预留，暂不开发
│   ├── gui/
│   ├── assets/
│   ├── cli.py
│   ├── entry.py
│   ├── build_dmg.sh
│   └── ...
│
├── tests/
└── README.md
```

---

## 3. 入口逻辑

`entry.py` 是整个应用的唯一入口点，PyInstaller 也只打包这一个入口。

```python
# linux/entry.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    if len(sys.argv) > 1:
        # 有参数 → CLI 模式
        from cli import run_cli
        run_cli(sys.argv[1:])
    else:
        # 无参数 → GUI 模式
        from gui.main_window import run_gui
        run_gui()


if __name__ == "__main__":
    main()
```

```
调用链：

  ./YoloLab.AppImage                    → entry.py → 无参数 → GUI
  ./YoloLab.AppImage train --epochs 50  → entry.py → 有参数 → cli.py
  yolo-lab tools split --ratio 0.8      → entry.py → 有参数 → cli.py
```

---

## 4. CLI 设计

### 4.1 命令体系

```
yolo-lab
├── train      训练模型
├── infer      推理/预测
├── tools      数据集工具
│   ├── split        切分数据集（train/val）
│   ├── labels       创建空标签文件
│   └── stats        数据集统计信息
└── config      配置文件管理
    ├── init          交互式创建配置文件
    ├── show          显示当前配置
    └── set           修改单个配置项
```

### 4.2 参数示例

```bash
# 训练
yolo-lab train --epochs 200 --batch 8
yolo-lab train --data ./my_data.yaml --model yolo11n-seg.pt

# 推理
yolo-lab infer --source ./image.jpg --model ./best.pt
yolo-lab infer --source ./images/ --model ./best.pt --conf 0.5

# 数据集工具
yolo-lab tools split --source ./all_images --ratio 0.8
yolo-lab tools labels --image-dir ./images/train
yolo-lab tools stats --data ./dataset.yaml

# 配置管理
yolo-lab config init                    # 交互式创建配置
yolo-lab config show                    # 打印当前配置
yolo-lab config set data_yaml ./my.yaml # 修改单项
```

### 4.3 argparse 实现结构

```python
# linux/cli.py
import argparse

def run_cli(argv):
    parser = argparse.ArgumentParser(prog="yolo-lab")
    sub = parser.add_subparsers(dest="command")

    # train 子命令
    p_train = sub.add_parser("train")
    p_train.add_argument("--epochs", type=int)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--data")
    p_train.add_argument("--model")
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--no-augment", action="store_true")
    # ... 其余训练参数

    # infer 子命令
    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--source", required=True)
    p_infer.add_argument("--model")
    p_infer.add_argument("--conf", type=float, default=0.25)
    # ...

    # tools 子命令
    p_tools = sub.add_parser("tools")
    tools_sub = p_tools.add_subparsers(dest="tool_cmd")
    p_split = tools_sub.add_parser("split")
    p_split.add_argument("--source", required=True)
    p_split.add_argument("--ratio", type=float, default=0.8)
    # ...

    # config 子命令
    p_cfg = sub.add_parser("config")
    cfg_sub = p_cfg.add_subparsers(dest="cfg_cmd")
    cfg_sub.add_parser("init")
    cfg_sub.add_parser("show")
    p_set = cfg_sub.add_parser("set")
    p_set.add_argument("key")
    p_set.add_argument("value")
    # ...

    args = parser.parse_args(argv)
    dispatch(args)
```

---

## 5. 配置系统

### 5.1 三层优先级

```
CLI 参数  >  配置文件  >  代码默认值
  (最高)        (中等)        (兜底)
```

### 5.2 配置文件

路径：`~/.yolo-lab/config.yaml`（用户家目录下，跨项目共享）

```yaml
# ~/.yolo-lab/config.yaml
data_yaml: /home/user/datasets/project/data.yaml
model_file: yolo11n-seg.pt
results_dir: /home/user/yolo_results
log_dir: /home/user/yolo_results/logs

epochs: 150
imgsz: 640
batch: 16
device: auto          # auto = 自动检测 GPU

# 数据增强（可选，不写用默认值）
augment:
  mosaic: 1.0
  fliplr: 0.5
```

### 5.3 配置合并逻辑

```python
# linux/config.py
def load_effective_config(cli_args):
    """合并 配置文件 + CLI 参数，返回最终 TrainConfig"""
    cfg = TrainConfig()                          # 默认值
    file_cfg = load_user_config()                # ~/.yolo-lab/config.yaml
    if file_cfg:
        cfg = merge(cfg, file_cfg)
    if cli_args:
        cfg = merge(cfg, vars(cli_args))         # CLI 参数覆盖
    return cfg
```

### 5.4 交互式初始化（`yolo-lab config init`）

```
$ yolo-lab config init

  ═══ YOLO-LAB 初始配置 ═══
  数据集路径 [./data.yaml]:
  默认模型 [yolo11n-seg.pt]:
  输出目录 [./results]:
  训练轮数 [150]:
  批次大小 [16]:
  GPU 自动检测？ [Y/n]: y

  ✓ 配置已保存到 ~/.yolo-lab/config.yaml
  运行 yolo-lab 或 yolo-lab train 开始使用。
```

---

## 6. GUI 设计（Linux）

从现有 `gui/` 代码整理而来，保持 PySide6 + 四 Tab 结构：

| Tab | 功能 | 来源 |
|-----|------|------|
| **训练** | 模型训练、参数配置、进度条 | 现有 `train_tab.py` |
| **推理** | 图片/目录推理、结果展示 | 现有 `infer_tab.py` |
| **工具** | 数据集切分、空标签创建 | 现有 `tools_tab.py` |
| **日志** | 训练/推理日志查看 | 现有 `log_viewer_tab.py` |

保留功能：GPU 自动检测、国际化（zh/en/fr/es）、首次使用引导。

---

## 7. 打包与安装（Linux）

### 7.1 AppImage 构建

```bash
$ cd linux/
$ ./build.sh --version 0.2.0
# 输出：dist/YoloLab-0.2.0-x86_64.AppImage
```

PyInstaller spec 关键配置：
- 入口：`linux/entry.py`
- 图标：`linux/assets/icon.png`
- `console=False`（GUI 模式不弹终端）
- `argv_emulation=True`（参数透传给 CLI）

### 7.2 安装脚本

```bash
$ sudo ./install.sh
# 1. 复制 AppImage 到 /opt/YoloLab/
# 2. 符号链接到 /usr/local/bin/yolo-lab
# 3. 安装 yolo_lab.desktop 到系统
```

安装后用户体验：

```bash
# 桌面：点击 YOLO-LAB 图标 → 启动 GUI
# 终端：
$ yolo-lab                           # 启动 GUI
$ yolo-lab train --epochs 100        # CLI 训练
```

### 7.3 Desktop 文件

```ini
# linux/yolo_lab.desktop
[Desktop Entry]
Name=YOLO-LAB
Comment=YOLO Segmentation Training & Inference Toolkit
Exec=/opt/YoloLab/YoloLab.AppImage
Icon=/opt/YoloLab/icon.png
Type=Application
Categories=Science;ArtificialIntelligence;
Terminal=false
```

---

## 8. 迁移计划

### 从现有代码迁移

| 现有位置 | 目标位置 | 说明 |
|----------|----------|------|
| `shared/config.py` | `linux/config.py` | TrainConfig 移到 Linux |
| `shared/train_core.py` | `linux/training.py` | 训练逻辑移到 Linux |
| `shared/train_logger.py` | `linux/train_logger.py` | 日志模块 |
| `shared/i18n_helper.py` | `linux/i18n.py` | i18n 合并 |
| `shared/env_check.py` | `linux/env_check.py` 或内置到 entry | 环境检查 |
| `gui/**` | `linux/gui/**` | GUI 整体搬迁 |
| `cli/**` | 整合到 `linux/cli.py` + `linux/training.py` | CLI 脚本重构为子命令 |
| `packaging/assets/` | `linux/assets/` | 图标按系统拆分 |
| `packaging/linux/` | `linux/build.sh` | 构建脚本整理 |

### 删除

| 删除项 | 原因 |
|--------|------|
| `shared/` | 不再跨平台共享 |
| `gui/`（根目录） | 迁入 `linux/gui/` |
| `cli/`（根目录） | 迁入 `linux/cli.py` |
| `packaging/` | 各系统自带构建脚本 |
| `.github/workflows/release.yml` 现有内容 | 按新结构重写 |

---

## 9. 开发阶段

| 阶段 | 内容 | 产出 |
|------|------|------|
| **1. 骨架搭建** | 创建 `linux/` 目录，`entry.py` + `cli.py` 框架 + `config.py` 配置系统 | 可运行的空壳 |
| **2. 核心迁移** | `shared/` 训练逻辑 → `linux/training.py`，`cli/` 脚本 → `linux/cli.py` 子命令 | CLI 功能可用 |
| **3. GUI 搬迁** | `gui/` → `linux/gui/`，修复导入路径 | GUI 功能可用 |
| **4. 打包** | `build.sh` + AppImage + `install.sh` + desktop 文件 | 可分发的安装包 |
| **5. 清理** | 删除 `shared/`、`gui/`、`cli/` 根目录旧文件，更新 CI | 干净仓库 |
| **6. 文档** | 重写 README，反映新架构 | 用户可读文档 |
| **7. 占位** | 创建 `windows/`、`macos/` 空壳（含 README 说明预留） | 结构完整 |

---

## 10. 后续扩展

| 系统 | 工作量 | 说明 |
|------|--------|------|
| **Windows** | 中等 | GUI 用 PySide6（一致），打包用 NSIS，入口逻辑相同 |
| **macOS** | 中等 | GUI 用 PySide6，打包用 DMG，注意 .app bundle 结构 |

两个系统的 `cli.py`、`training.py`、`config.py` 可直接从 Linux 复制，GUI 部分可视平台特性调整。

---

## 11. 风险与约束

- **回退点**：当前 commit `95e392c` 是已知稳定的回退点，迁移过程中随时可回滚
- **不破坏现有功能**：迁移 = 搬文件 + 修正导入路径，不重写业务逻辑
- **测试**：迁移后原有测试需适配新导入路径
