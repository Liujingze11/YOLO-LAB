# YOLO-LAB

YOLO 模型训练与推理工具集，提供图形界面 (GUI) 和命令行 (CLI) 两种使用方式。

---

## 目录

| 子项目 | 说明 |
|--------|------|
| `gui/` | PySide6 图形界面，支持训练/推理/数据集工具 |
| `cli/` | 命令行工具，功能等效于 GUI |

---

## 快速开始

### 从源码运行

```bash
git clone https://github.com/Liujingze11/YOLO-LAB.git
cd YOLO-LAB

# 安装依赖（推荐使用 Conda）
conda create -n yolo python=3.10 -y
conda activate yolo
pip install -r gui/requirements.txt

# 启动 GUI
python gui/gui/main.py
```

### 下载安装包（推荐）

前往 [Releases](https://github.com/Liujingze11/YOLO-LAB/releases) 下载对应平台安装包：

| 平台 | 安装包 | 说明 |
|------|--------|------|
| **Windows** | `YoloLab-x.x.x-Setup.exe` | 双击安装，可选 GPU 加速组件 |
| **Linux** | `YoloLab-x.x.x-x86_64.AppImage` | 下载后 `chmod +x` 即可运行 |
| **macOS** | `YoloLab-x.x.x-arm64.dmg` | 拖入 Applications 使用 |

> 默认安装包为 CPU 版本（~1.3GB），不含 GPU 加速。如需 GPU 训练，参见下方 [GPU 加速设置](#gpu-加速设置)。

---

## GPU 加速设置

### 自动检测与下载

1. 打开应用，进入 **训练** 标签页
2. 在 **Device** 下拉框中选择 GPU 选项（显示"点击检测"）
3. 应用自动检测 NVIDIA 驱动：
   - **检测到 GPU** → 弹窗提示下载 CUDA 组件（约 2.5GB），确认后自动下载
   - **未检测到** → 提示当前设备不支持，使用 CPU 训练
4. 下载完成后**重启应用**，GPU 即可使用

### 各平台 GPU 支持

| 平台 | GPU 方案 | 需要下载 |
|------|---------|---------|
| Windows / Linux（NVIDIA） | CUDA | 是（~2.5GB） |
| macOS（Apple Silicon） | MPS (Metal) | **不需要**，CPU 版直接支持 |
| AMD 显卡 | 暂不支持 | — |

### 手动安装 GPU 组件

如果自动下载失败，可以手动操作：

1. 访问 [PyTorch 官网](https://pytorch.org) 选择 CUDA 版本
2. 将下载的 wheel 文件安装到用户数据目录：
   - **Windows**: `%APPDATA%\YoloLab\gpu\`
   - **Linux**: `~/.local/share/YoloLab/gpu/`
   - **macOS**: `~/Library/Application Support/YoloLab/gpu/`
3. 在该目录创建 `gpu_ready.json`（内容：`{"cuda_version": "cu121"}`）
4. 重启应用

---

## 本地构建

### 环境要求
- Python 3.10+
- pip

### Linux / macOS

```bash
# CPU 版本（默认，~1.3GB）
./packaging/build_local.sh --version 0.1.0

# 同时构建 CUDA 组件（额外生成 gpu_bundle.zip）
./packaging/build_local.sh --version 0.1.0 --cuda
```

### Windows

```bat
:: CPU 版本
.\packaging\build_local.bat --version 0.1.0

:: 同时构建 CUDA 组件
.\packaging\build_local.bat --version 0.1.0 --cuda
```

### 发布自动构建

推送版本 tag 触发 GitHub Actions 自动构建：

```bash
git tag v0.1.0
git push origin v0.1.0
```

CI 将自动构建 Windows/Linux/macOS 安装包并发布到 Release 页面。

---

## 用户数据目录

应用运行时数据存储在系统标准路径：

| 平台 | 路径 |
|------|------|
| Windows | `%APPDATA%\YoloLab\` |
| Linux | `~/.local/share/YoloLab/` |
| macOS | `~/Library/Application Support/YoloLab/` |

包含：模型权重 (`models/`)、训练结果 (`results/`)、日志 (`logs/`)、推理结果 (`predict/`)、GPU 组件 (`gpu/`)。
