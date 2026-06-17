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

前往 [Releases](https://github.com/Liujingze11/YOLO-LAB/releases) 下载 `YoloLab.AppImage`，然后：

```bash
chmod +x YoloLab-*.AppImage
sudo ./linux/install.sh    # 一键安装到系统

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
yolo-lab config set data_yaml /path/to/dataset.yaml
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

---

## 用户数据目录

| 平台 | 路径 |
|------|------|
| Windows | `%APPDATA%\YoloLab\` |
| Linux | `~/.local/share/YoloLab/` |
| macOS | `~/Library/Application Support/YoloLab/` |
