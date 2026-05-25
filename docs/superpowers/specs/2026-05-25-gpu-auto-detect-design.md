# GPU 能力检测与自动下载设计

**日期**: 2026-05-25
**目标**: 应用默认发布 CPU 版（~600MB），用户选择 GPU 设备时自动检测 CUDA 兼容性并引导下载

---

## 1. 整体流程

```
用户选择 GPU 设备
        │
        ▼
  torch.cuda.is_available()?
        │
    ┌───┴───┐
   是        否
    │         │
    ▼         ▼
 直接使用  检测 NVIDIA 驱动
           (nvidia-smi)
            │
        ┌───┴───┐
       有        无
        │         │
        ▼         ▼
   显示下载确认框   提示不支持并切回 CPU
   ~2.5GB 确认
   [确认] [取消]
        │
   用户确认?
    │    │
   是   否
    │    │
    ▼    ▼
 下载安装  切回 CPU
    │
    ▼
 提示重启应用
```

### 平台差异

| 场景 | 行为 |
|------|------|
| **CUDA 已安装** | 直接使用，不做任何提示 |
| **NVIDIA 驱动检测到，无 CUDA torch** | 显示下载确认框，展示 GPU 型号 |
| **无 NVIDIA 驱动** | "未检测到 NVIDIA 显卡或 CUDA 驱动，当前版本不支持您的设备，将使用 CPU" |
| **macOS（Apple Silicon）** | 自动使用 MPS，无需下载 |
| **macOS（Intel）** | 无 GPU 选项，仅 CPU |
| **AMD GPU** | 检测到 rocm-smi 时提示"暂不支持 AMD GPU，将使用 CPU" |

---

## 2. 下载与安装

### 下载源（按优先级）
1. GitHub Release 附件 `gpu_bundle.zip`（由 CI 构建，最快）
2. PyTorch 官方源 (`download.pytorch.org/whl/cu121`)

### 安装流程
```
1. GUI 显示下载进度对话框（复用 DownloadDialog）
2. 下载 gpu_bundle.zip 或 PyTorch CUDA wheel
3. 解压/安装到 {用户数据目录}/gpu/
4. 写入 gpu_ready.json 标记文件
5. 弹窗"安装完成，请重启应用"
6. 用户重启 → 检测 gpu_ready.json → sys.path 插入 → torch.cuda 可用
```

### 备用方案
网络故障时弹窗提示手动安装：
```
"自动下载失败。请手动访问 https://pytorch.org 下载 CUDA 12.1 版本，
安装到 {用户数据目录}/gpu/ 目录后重启应用"
```

---

## 3. 启动加载逻辑

```python
# 应用启动时
def load_torch():
    gpu_dir = get_user_data_dir() / "gpu"
    gpu_ready = (gpu_dir / "gpu_ready.json").is_file()
    if gpu_ready:
        sys.path.insert(0, str(gpu_dir))
    import torch
    return torch
```

---

## 4. 文件改动

### 新增
| 文件 | 职责 |
|------|------|
| `gui/gui/gpu_manager.py` | GPU 检测、CUDA 兼容性检查、下载管理、安装逻辑 |

### 修改
| 文件 | 改动 |
|------|------|
| `gui/gui/device.py` | `get_available_devices()` 增加 GPU 可用性检测，返回元数据（型号、是否需下载） |
| `gui/gui/main.py` | Device 下拉框选中 GPU 时触发 `GpuManager.check_and_prompt()` |
| `gui/gui/model_selector.py` | `DownloadDialog` 改为可复用组件（GPU 下载共用） |
| `gui/gui/paths.py` | `ensure_user_dirs()` 增加 `gpu` 子目录 |
| `gui/locales/*.json` | 新增 GPU 提示文本翻译 key（约 15 个） |

### 打包
| 文件 | 改动 |
|------|------|
| `packaging/yolo_lab.spec` | 默认排除 `nvidia`、`triton`、`libnvJitLink` 等 CUDA 库 |
| `packaging/build_local.sh` | `--cuda` 改为额外构建 `gpu_bundle.zip` 而非打包进主程序 |
| `packaging/build_local.bat` | 同上 |
| `.github/workflows/release.yml` | 新增 job：构建并上传 `gpu_bundle.zip` 到 Release 附件 |
| `packaging/windows/setup.iss` | 安装向导中增加可选勾选"下载 CUDA 加速组件" |

---

## 5. GPU 组件版本管理

- `gpu_ready.json` 记录已安装的 CUDA 版本：`{"cuda_version": "cu121", "torch_version": "2.5.1"}`
- 应用启动时检测版本兼容性，不匹配时提示重新下载
- 应用更新时 GPU 组件不受影响（在用户数据目录）

---

## 6. 用户体验

- CPU 版安装包 ~600MB，启动秒开
- GPU 检测在后台执行，不影响启动速度
- 下载过程有进度条和取消按钮
- 所有提示文案支持 4 种语言（zh/en/fr/es）
