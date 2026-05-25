# GPU Auto-Detect & Download Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Default CPU-only build (~600MB), detect NVIDIA GPU on device selection, prompt user to download CUDA components (~2.5GB) with progress dialog.

**Architecture:** New `gpu_manager.py` handles all detection/download/install logic. `device.py` returns GPU capabilities metadata (not just names). `main.py` triggers the check when user selects GPU in the device combobox. CUDA components install to user data dir, loaded on next app restart via `sys.path` injection.

**Tech Stack:** Python, PySide6 QThread, urllib, subprocess (nvidia-smi)

---

### Task 1: Add GPU directory to user data paths

**Files:**
- Modify: `gui/gui/paths.py`

- [ ] **Step 1: Add `gpu` to `ensure_user_dirs()`**

Read `gui/gui/paths.py`. In the `ensure_user_dirs()` function, add `"gpu"` entry:

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add gui/gui/paths.py
git commit -m "feat: add gpu directory to user data dirs"
```

---

### Task 2: Add GPU translation keys to locales

**Files:**
- Modify: `gui/locales/zh.json`
- Modify: `gui/locales/en.json`

- [ ] **Step 1: Add keys to zh.json**

Add these keys to the end of the JSON object (before the closing `}`):

```json
    "gpu.check_title": "GPU 加速检测",
    "gpu.checking": "正在检测 GPU 兼容性...",
    "gpu.download_title": "CUDA 组件下载",
    "gpu.download_prompt": "检测到 {name}\n需要下载 CUDA 加速组件才能使用 GPU 训练，约 {size} GB。\n是否下载？",
    "gpu.downloading": "正在下载 CUDA 组件...",
    "gpu.download_done": "下载完成！请重启应用以启用 GPU 加速。",
    "gpu.download_failed": "下载失败",
    "gpu.no_nvidia": "未检测到 NVIDIA 显卡或 CUDA 驱动，当前版本不支持您的设备，将使用 CPU 进行训练。",
    "gpu.no_amd": "暂不支持 AMD GPU，将使用 CPU 进行训练。",
    "gpu.install_manual": "自动下载失败。请手动访问 https://pytorch.org 下载 CUDA 12.1 版本，安装后重启应用。",
    "gpu.restart_needed": "安装完成，请重启应用",
    "gpu.device_cpu": "CPU",
    "gpu.device_gpu_download": "GPU (需下载 CUDA 组件)"
```

- [ ] **Step 2: Add keys to en.json**

```json
    "gpu.check_title": "GPU Acceleration Check",
    "gpu.checking": "Checking GPU compatibility...",
    "gpu.download_title": "CUDA Component Download",
    "gpu.download_prompt": "Detected {name}\nCUDA acceleration components (~{size} GB) need to be downloaded to use GPU training.\nProceed with download?",
    "gpu.downloading": "Downloading CUDA components...",
    "gpu.download_done": "Download complete! Please restart the app to enable GPU acceleration.",
    "gpu.download_failed": "Download failed",
    "gpu.no_nvidia": "No NVIDIA GPU or CUDA driver detected. This version does not support your device. Training will use CPU.",
    "gpu.no_amd": "AMD GPU is not currently supported. Training will use CPU.",
    "gpu.install_manual": "Automatic download failed. Please visit https://pytorch.org to download CUDA 12.1, install it, then restart the app.",
    "gpu.restart_needed": "Installation complete, please restart the app",
    "gpu.device_cpu": "CPU",
    "gpu.device_gpu_download": "GPU (CUDA download required)"
```

- [ ] **Step 3: Add keys to fr.json and es.json**

Add the same keys with French/Spanish translations (use the en.json values as fallback for now — add translated variants later).

- [ ] **Step 4: Commit**

```bash
git add gui/locales/
git commit -m "feat: add GPU detection i18n keys"
```

---

### Task 3: Create gpu_manager.py

**Files:**
- Create: `gui/gui/gpu_manager.py`

- [ ] **Step 1: Write gpu_manager.py**

```python
"""GPU detection, CUDA compatibility check, and download management."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from PySide6.QtCore import QThread, Signal, Slot

from gui.paths import get_user_data_dir, is_frozen
from gui.i18n import tr


def get_gpu_dir() -> Path:
    return get_user_data_dir() / "gpu"


def get_gpu_ready_file() -> Path:
    return get_gpu_dir() / "gpu_ready.json"


def is_cuda_available() -> bool:
    """Check if CUDA torch is already available (via import or gpu_ready)."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def has_gpu_ready_marker() -> bool:
    return get_gpu_ready_file().is_file()


def _check_nvidia_driver() -> tuple[bool, str]:
    """Check if NVIDIA driver is present. Returns (has_driver, gpu_name)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            name = result.stdout.strip().split("\n")[0]
            return True, name
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False, ""


def _check_amd_driver() -> bool:
    """Check if AMD GPU driver is present."""
    try:
        result = subprocess.run(
            ["rocm-smi"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_gpu_capability() -> dict:
    """Check system GPU capability. Returns status dict.

    Returns:
        {
            "status": "cuda_ready" | "needs_download" | "no_nvidia" | "has_amd" | "macos_mps" | "cpu_only",
            "gpu_name": str or None,
            "download_size_gb": float or None,
            "message": str (translated user-facing message),
        }
    """
    system = platform.system()

    # macOS
    if system == "Darwin":
        import torch
        try:
            if torch.backends.mps.is_available():
                return {"status": "macos_mps", "gpu_name": "Apple GPU (MPS)", "download_size_gb": None, "message": ""}
        except Exception:
            pass
        return {"status": "cpu_only", "gpu_name": None, "download_size_gb": None, "message": ""}

    # Already has CUDA
    if is_cuda_available():
        import torch
        name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "NVIDIA GPU"
        return {"status": "cuda_ready", "gpu_name": name, "download_size_gb": None, "message": ""}

    # Check NVIDIA driver
    has_nvidia, gpu_name = _check_nvidia_driver()
    if has_nvidia:
        return {
            "status": "needs_download",
            "gpu_name": gpu_name,
            "download_size_gb": 2.5,
            "message": tr("gpu.download_prompt", name=gpu_name, size="2.5"),
        }

    # Check AMD
    if _check_amd_driver():
        return {"status": "has_amd", "gpu_name": None, "download_size_gb": None, "message": tr("gpu.no_amd")}

    # No GPU
    return {"status": "no_nvidia", "gpu_name": None, "download_size_gb": None, "message": tr("gpu.no_nvidia")}


class GpuDownloadWorker(QThread):
    """Background thread for downloading CUDA components."""
    progress = Signal(int, int)  # current, total (bytes)
    finished = Signal(bool, str)  # success, message
    error_msg = Signal(str)

    def __init__(self, download_url: str, dest_dir: Path):
        super().__init__()
        self._url = download_url
        self._dest = dest_dir / "gpu_bundle.zip"
        self._tmp = dest_dir / "gpu_bundle.zip.tmp"
        self._aborted = False

    def run(self) -> None:
        try:
            self._dest.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(self._url) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                with open(self._tmp, "wb") as f:
                    while True:
                        if self._aborted:
                            self._cleanup()
                            return
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        self.progress.emit(downloaded, total)
            if self._aborted:
                self._cleanup()
                return
            os.replace(str(self._tmp), str(self._dest))
            self._install_bundle()
            self.finished.emit(True, self._dest.name)
        except Exception as exc:
            self._cleanup()
            self.error_msg.emit(str(exc))
            self.finished.emit(False, str(exc))

    def _install_bundle(self) -> None:
        """Extract gpu_bundle.zip into the gpu directory."""
        import zipfile
        dest_dir = self._dest.parent
        with zipfile.ZipFile(self._dest, "r") as zf:
            zf.extractall(dest_dir)
        # Write ready marker
        ready = {"cuda_version": "cu121", "installed_at": __import__("datetime").datetime.now().isoformat()}
        with open(dest_dir / "gpu_ready.json", "w") as f:
            json.dump(ready, f)
        # Clean up zip
        self._dest.unlink(missing_ok=True)

    def stop(self) -> None:
        self._aborted = True

    def _cleanup(self) -> None:
        if self._tmp.is_file():
            self._tmp.unlink(missing_ok=True)
```

- [ ] **Step 2: Commit**

```bash
git add gui/gui/gpu_manager.py
git commit -m "feat: add GPU detection and download manager"
```

---

### Task 4: Update device.py with GPU capability metadata

**Files:**
- Modify: `gui/gui/device.py`

- [ ] **Step 1: Replace device.py**

Read `gui/gui/device.py` and replace the entire file with:

```python
"""设备自动检测 — 优先 GPU，无 GPU 则回退 CPU。"""
from __future__ import annotations

from gui.gpu_manager import is_cuda_available, has_gpu_ready_marker


def get_default_device() -> str:
    """返回最佳可用设备 ID：'0'（第一块 GPU）→ 'cpu'。"""
    try:
        if is_cuda_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def get_available_devices() -> list[tuple[str, str]]:
    """返回可用设备列表：(device_id, display_name)。"""
    devices: list[tuple[str, str]] = [("cpu", "CPU")]
    try:
        import torch
        if is_cuda_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                total_mem = torch.cuda.get_device_properties(i).total_memory
                mem_gb = total_mem / (1024 ** 3)
                devices.append((str(i), f"GPU {i}: {name} ({mem_gb:.1f}G)"))
        else:
            # CUDA not installed but user data might have it (needs restart)
            if has_gpu_ready_marker():
                devices.append(("gpu", "GPU (需重启应用以启用)"))
            else:
                # Show GPU option to trigger download
                import platform
                if platform.system() != "Darwin":
                    devices.append(("gpu", "GPU (点击检测)"))
                else:
                    # macOS: check MPS
                    try:
                        if torch.backends.mps.is_available():
                            devices.append(("mps", "Apple GPU (MPS)"))
                    except Exception:
                        pass
    except Exception:
        pass
    return devices
```

- [ ] **Step 2: Commit**

```bash
git add gui/gui/device.py
git commit -m "feat: add GPU capability metadata to device detection"
```

---

### Task 5: Update main.py Device combobox to trigger GPU check

**Files:**
- Modify: `gui/gui/main.py`

- [ ] **Step 1: Add GPU check method to MainWindow**

Read `gui/gui/main.py`. Add this method to the `MainWindow` class (e.g., after `_refresh_devices`):

```python
    def _on_device_selected(self, idx: int) -> None:
        """Trigger GPU capability check when user selects a GPU device."""
        device_id = self.tr_device.itemData(idx)
        if device_id != "gpu":
            return

        from gui.gpu_manager import check_gpu_capability, GpuDownloadWorker, get_gpu_dir
        from PySide6.QtWidgets import QMessageBox

        capability = check_gpu_capability()

        if capability["status"] == "needs_download":
            reply = QMessageBox.question(
                self, tr("gpu.download_title"),
                capability["message"],
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                self.tr_device.setCurrentIndex(0)  # switch back to CPU
                return

            # Start download
            self._gpu_download_dialog = None
            from gui.model_selector import DownloadDialog
            self._gpu_download_dialog = DownloadDialog("CUDA components", self.window())
            self._gpu_download_dialog.show()

            RELEASE_URL = "https://github.com/user/yolo_lab_gui/releases/latest/download/gpu_bundle.zip"
            self._gpu_worker = GpuDownloadWorker(RELEASE_URL, get_gpu_dir())
            self._gpu_worker.progress.connect(self._gpu_download_dialog.set_progress)
            self._gpu_worker.finished.connect(self._on_gpu_download_done)
            self._gpu_worker.error_msg.connect(self._on_gpu_download_error)
            self._gpu_worker.start()

        elif capability["status"] in ("no_nvidia", "has_amd"):
            QMessageBox.information(self, tr("gpu.check_title"), capability["message"])
            self.tr_device.setCurrentIndex(0)

        elif capability["status"] == "macos_mps":
            # macOS MPS is available, select it
            pass  # The MPS device is already listed separately

    def _on_gpu_download_done(self, success: bool, msg: str) -> None:
        if self._gpu_download_dialog:
            self._gpu_download_dialog.accept()
            self._gpu_download_dialog = None
        from PySide6.QtWidgets import QMessageBox
        if success:
            QMessageBox.information(self, tr("gpu.download_title"), tr("gpu.restart_needed"))
        else:
            QMessageBox.critical(self, tr("gpu.download_failed"), msg[:500])
            QMessageBox.information(self, tr("gpu.download_title"), tr("gpu.install_manual"))
        self.tr_device.setCurrentIndex(0)  # switch back to CPU
        self._refresh_devices()

    def _on_gpu_download_error(self, msg: str) -> None:
        if self._gpu_download_dialog:
            self._gpu_download_dialog.set_error(msg)
```

- [ ] **Step 2: Wire up the signal**

Find where `self.tr_device` combo is created (around line 231 in `_build_train_tab`) and add the signal connection after its creation:

```python
        self.tr_device.currentIndexChanged.connect(self._on_device_selected)
```

Note: There might already be a `currentIndexChanged` connection. If so, the `_on_device_selected` logic can be integrated into the existing handler or connected as a separate slot.

- [ ] **Step 3: Commit**

```bash
git add gui/gui/main.py
git commit -m "feat: trigger GPU detection on device selection"
```

---

### Task 6: Update packaging — exclude CUDA libs from default build

**Files:**
- Modify: `packaging/yolo_lab.spec`

- [ ] **Step 1: Add CUDA-related excludes to spec**

Read `packaging/yolo_lab.spec`. Add these to the `excludes` list:

```python
    excludes=[
        "torchaudio",
        # CUDA libraries — excluded from default build, available via GPU bundle
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
        ...
    ],
```

Also remove `"nvidia.cudnn"` and `"nvidia.nccl"` and `"nvidia.nvshmem"` from `hiddenimports` if they're listed there (they were auto-added by the torch hook).

- [ ] **Step 2: Add binaries exclude for CUDA DLLs**

Add to `binaries` in Analysis:
```python
    binaries=[
        (str(Path(sys.prefix) / "lib" / "libexpat.so.1"), "."),
    ],
    hookspath=[],
    hooksconfig={},
    ...
```

- [ ] **Step 3: Commit**

```bash
git add packaging/yolo_lab.spec
git commit -m "feat: exclude CUDA libs from default build"
```

---

### Task 7: Update local build scripts — GPU bundle

**Files:**
- Modify: `packaging/build_local.sh`
- Modify: `packaging/build_local.bat`

- [ ] **Step 1: Add GPU bundle build to build_local.sh**

After the PyInstaller build section, add:

```bash
# GPU bundle (optional)
if [ "$CUDA" = "1" ]; then
    echo "=== Building GPU bundle ==="
    GPU_DIR="$SCRIPT_DIR/dist/gpu_bundle"
    rm -rf "$GPU_DIR"
    mkdir -p "$GPU_DIR"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
        --target "$GPU_DIR" --no-deps
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
        --target "$GPU_DIR"
    # Create zip
    cd "$SCRIPT_DIR/dist"
    zip -r "gpu_bundle.zip" "gpu_bundle/"
    rm -rf "gpu_bundle/"
    echo "GPU bundle: $SCRIPT_DIR/dist/gpu_bundle.zip"
fi
```

- [ ] **Step 2: Add to build_local.bat**

```bat
if "%CUDA%"=="1" (
    echo === Building GPU bundle ===
    set GPU_DIR=%SCRIPT_DIR%dist\gpu_bundle
    if exist "%GPU_DIR%" rmdir /s /q "%GPU_DIR%"
    mkdir "%GPU_DIR%"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --target "%GPU_DIR%"
    powershell Compress-Archive -Path "%GPU_DIR%\*" -DestinationPath "%SCRIPT_DIR%dist\gpu_bundle.zip"
    rmdir /s /q "%GPU_DIR%"
    echo GPU bundle: %SCRIPT_DIR%dist\gpu_bundle.zip
)
```

- [ ] **Step 3: Commit**

```bash
git add packaging/build_local.sh packaging/build_local.bat
git commit -m "feat: add GPU bundle build to local scripts"
```

---

### Task 8: Update CI workflow — GPU bundle release artifact

**Files:**
- Modify: `.github/workflows/release.yml`

- [ ] **Step 1: Add GPU bundle build job**

Add this job to the workflow (runs in parallel with existing build jobs):

```yaml
  build-gpu-bundle:
    runs-on: ubuntu-latest
    if: ${{ inputs.cuda }}
    steps:
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Build GPU bundle
        run: |
          pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
            --target gpu_bundle
          zip -r gpu_bundle.zip gpu_bundle/

      - name: Upload GPU bundle
        uses: actions/upload-artifact@v4
        with:
          name: gpu_bundle.zip
          path: gpu_bundle.zip
```

- [ ] **Step 2: Add gpu_bundle.zip to release job artifacts**

In the `release` job, the `download-artifact@v4` step with `path: artifacts` will already pick up the GPU bundle. No further changes needed.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "feat: add GPU bundle build to CI workflow"
```

---

### Task 9: Update Windows installer — optional CUDA component

**Files:**
- Modify: `packaging/windows/setup.iss`

- [ ] **Step 1: Add optional CUDA component to setup.iss**

Add a components section and a task for optional GPU bundle download/install:

```ini
[Components]
Name: "cuda"; Description: "CUDA GPU acceleration support (requires NVIDIA GPU)"; Types: full custom; Flags: disablenouninstallwarning

[Files]
Source: "..\dist\gpu_bundle.zip"; DestDir: "{userappdata}\YoloLab\gpu"; Components: cuda; Flags: ignoreversion

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
```

- [ ] **Step 2: Commit**

```bash
git add packaging/windows/setup.iss
git commit -m "feat: add optional CUDA component to Windows installer"
```

---

### Task 10: Rebuild CPU-only version and test

**Files:**
- Test: `dist/YoloLab/` (rebuild)

- [ ] **Step 1: Install CPU torch and rebuild**

```bash
source /home/ljz/miniconda3/etc/profile.d/conda.sh && conda activate yolo
pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall
pyinstaller --clean --noconfirm packaging/yolo_lab.spec
```

- [ ] **Step 2: Verify build size**

```bash
du -sh dist/YoloLab/
# Expected: ~600-800MB (not 4.8GB)
```

- [ ] **Step 3: Verify app launches without CUDA errors**

```bash
timeout 5 dist/YoloLab/YoloLab 2>&1; echo "Exit: $?"
# Expected: no traceback, exit 124 (killed by timeout = running normally)
```

- [ ] **Step 4: Verify GPU option appears in device list**

Launch the app and check the Device dropdown shows: CPU, "GPU (点击检测)"

- [ ] **Step 5: Commit any fixes**

```bash
git add -A && git commit -m "fix: CPU build verification adjustments"
```

---

### Post-Implementation Notes

**How users get GPU support:**
1. They download CPU-only installer (~200-300MB compressed)
2. Open app, go to Train tab, select GPU in Device dropdown
3. App detects NVIDIA GPU → shows download prompt
4. User confirms → downloads ~2.5GB GPU bundle
5. Restart app → GPU training available

**CI release flow:**
- `git tag v0.1.0` → builds CPU installers + GPU bundle
- User downloads `YoloLab-0.1.0-Setup.exe` + optionally `gpu_bundle.zip`
- App can auto-download `gpu_bundle.zip` from GitHub Release
