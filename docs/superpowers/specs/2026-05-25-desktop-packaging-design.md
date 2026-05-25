# YoloLab 桌面应用打包设计

**日期**: 2026-05-25
**目标**: 将 PySide6 GUI 应用封装为 Linux/macOS/Windows 三平台桌面安装程序

---

## 1. 方案选型

**方案 B: PyInstaller + 平台安装脚本**

| 平台 | 打包链 |
|------|--------|
| Windows | PyInstaller → Inno Setup → `.exe` 安装向导 |
| Linux | PyInstaller → AppImageTool → `.AppImage` + `.deb` |
| macOS | PyInstaller → create-dmg → `.dmg` |

选型理由: PyInstaller 在 PySide6+PyTorch 社区最成熟，排查问题资料丰富。

---

## 2. 目录结构（零侵入）

```
packaging/
├── yolo_lab.spec                 # PyInstaller spec（三平台共用）
├── assets/
│   ├── icon.png                  # 通用图标 512x512
│   ├── icon.ico                  # Windows 图标
│   └── icon.icns                 # macOS 图标
├── windows/
│   └── setup.iss                 # Inno Setup 脚本
├── linux/
│   └── AppImageBuilder.yml       # AppImage 配方
├── macos/
│   └── create_dmg.sh             # DMG 生成脚本
├── build_local.sh                # Linux/macOS 本地构建
└── build_local.bat               # Windows 本地构建
```

所有打包产物在 `packaging/` 下，零侵入源码。

---

## 3. 子进程 → 进程内调用

**改造**: `workers.py` 中的 `subprocess.Popen` 调用改为直接 import 引擎函数。

| 文件 | 改动 |
|------|------|
| `gui/workers.py` | 重写 `TrainWorker`、`InferWorker`、`ToolWorker` 的 `run()`，改为进程内调用引擎函数 |
| `gui/train_engine.py` | 抽取 `run_training(args, on_log, on_progress, check_cancel)` 函数 |
| `gui/infer_engine.py` | 抽取 `run_inference(args, on_log, on_progress, check_cancel)` 函数 |
| `gui/main.py` | cmd 列表构建改为 args 字典，传给 Worker |
| `tools/dataset_tools/*.py` | 工具脚本提供 `run_tool()` 函数入口 |

保留 scripts 的 argparse 入口，确保 CLI 项目仍可独立运行。

---

## 4. PyInstaller 配置

**入口点**: `gui/gui/main.py:main()`
**应用名**: `YoloLab`

**资源文件 (--add-data)**:
- `gui/locales/*.json` → `locales/`
- `gui/gui/infer_task_params.json` → `gui/`

**隐藏导入 (--hidden-import)**:
- `ultralytics`, `torch`, `cv2`, `numpy`, `yaml`
- `PySide6.QtCore`, `PySide6.QtGui`, `PySide6.QtWidgets`

**排除模块 (--exclude)**:
- `torchvision`, `torchaudio`
- `PySide6.QtWebEngine*`, `PySide6.QtMultimedia*`, `PySide6.QtBluetooth*`

**二进制优化**:
- UPX 压缩 .pyd/.so/.dll
- strip 调试符号

**打包产物**:
```
dist/YoloLab/
├── yolo_lab(.exe)               # 可执行文件
├── _internal/                    # Python 运行时 + 全部依赖
├── locales/                      # 翻译文件
└── gui/                          # 推理参数 JSON
```

---

## 5. 体积优化

**分版本发布**:
| 版本 | 优化后安装包 |
|------|-------------|
| CPU 版 (pytorch cpuonly) | ~200 MB |
| CUDA 版 (pytorch cu118) | ~1.0 GB |

**优化手段**:
- 排除 torchvision/torchaudio
- 裁剪 PySide6 无用模块 (QtWebEngine, QtMultimedia, QtBluetooth 等)
- UPX 压缩二进制
- pip --no-deps 控制依赖

---

## 6. 平台安装器

### Windows (Inno Setup)
- 安装向导（许可协议、路径选择）
- 开始菜单快捷方式 + 桌面图标（可选）
- 控制面板卸载入口
- 文件关联 `.pt` 权重文件
- 输出: `YoloLab-x.x.x-Setup.exe`

### Linux (AppImage)
- 单文件，跨发行版
- 双击即运行，无需 root
- 内置 .desktop 文件 → 桌面菜单集成
- 可选 .deb 包（Ubuntu/Debian 用户）
- 输出: `YoloLab-x.x.x-x86_64.AppImage`

### macOS (DMG)
- 拖入 Applications 文件夹安装
- 需 macOS 环境构建（CI 用 GitHub macOS runner）
- 输出: `YoloLab-x.x.x-arm64.dmg` 和 `x86_64.dmg`

---

## 7. 用户数据目录

安装目录只读，运行时数据写入系统标准路径：

| 数据 | 路径 |
|------|------|
| 模型权重 | `{DATA}/models/` |
| 训练结果 | `{DATA}/results/` |
| 训练日志 | `{DATA}/logs/` |
| 推理结果 | `{DATA}/predict/` |
| 预设配置 | `{DATA}/presets.json` |

`{DATA}` 按平台:
| 平台 | 路径 |
|------|------|
| Windows | `%APPDATA%\YoloLab\` |
| Linux | `~/.local/share/YoloLab/` |
| macOS | `~/Library/Application Support/YoloLab/` |

首次启动自动创建目录。用户可浏览选择自定义路径。

---

## 8. 版本管理

- 版本号由 git tag 驱动（如 `v0.1.0`、`v0.2.0`）
- 安装包自动命名: `YoloLab-0.1.0-Setup.exe` 等
- 应用内窗口标题显示版本号

---

## 9. CI/CD (GitHub Actions)

**触发**: 推送 `v*` tag

**工作流**:
```
v0.1.0 tag →
  ├── build-windows (windows-latest)
  │     └── 产出: YoloLab-0.1.0-Setup.exe
  ├── build-linux   (ubuntu-latest)
  │     └── 产出: YoloLab-0.1.0-x86_64.AppImage + .deb
  └── build-macos   (macos-latest)
        └── 产出: YoloLab-0.1.0-arm64.dmg + x86_64.dmg

全部完成 → 创建 GitHub Release → 上传所有安装包
```

默认构建 CPU 版，CUDA 版通过 workflow_dispatch 手动触发。

---

## 10. 本地构建

```bash
# Linux/macOS
./packaging/build_local.sh                      # CPU 版
./packaging/build_local.sh --cuda               # CUDA 版
./packaging/build_local.sh --version 0.1.0      # 指定版本

# Windows
.\packaging\build_local.bat                      # CPU 版
.\packaging\build_local.bat --cuda               # CUDA 版
```

自动检测当前平台，构建对应安装包，输出到 `packaging/dist/`。
