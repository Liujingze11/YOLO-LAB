# Desktop Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the PySide6 GUI app as native desktop installers for Windows, Linux, and macOS.

**Architecture:** Keep subprocess isolation for train/infer/tool tasks (preserving cancel capability), but have subprocesses call `sys.executable` with engine-mode flags. In PyInstaller builds, `sys.executable` points to the frozen exe, so the same binary runs in different modes. Worker threads launch the engine via subprocess, read stdout, and emit Qt signals — same as today, only the cmd construction changes.

**Tech Stack:** PyInstaller, Inno Setup (Windows), AppImageTool (Linux), create-dmg (macOS), GitHub Actions

---

### Task 1: Create packaging directory structure and icon

**Files:**
- Create: `packaging/assets/icon.png` (placeholder)
- Create: `packaging/assets/icon.ico` (placeholder)
- Create: `packaging/assets/icon.icns` (placeholder)

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p packaging/{assets,windows,linux,macos}
```

- [ ] **Step 2: Generate placeholder icon (solid color PNG)**

```bash
python3 -c "
import struct, zlib
# 64x64 blue square PNG
def create_png(path, size=64):
    def chunk(ctype, data):
        c = ctype + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    raw = b''
    for y in range(size):
        raw += b'\x00' + b'\x00\x44\xff' * size  # filter=0, RGBA blue
    ihdr = struct.pack('>IIBBBBB', size, size, 8, 6, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')
with open('packaging/assets/icon.png', 'wb') as f:
    f.write(create_png('packaging/assets/icon.png'))
print('icon.png created')
"
```

- [ ] **Step 3: Convert PNG to ICO using Python**

```bash
python3 -c "
# Minimal .ico file wrapping the PNG
import struct
png_data = open('packaging/assets/icon.png', 'rb').read()
# ICO header + 1 entry + PNG data
ico = struct.pack('<HHH', 0, 1, 1)  # reserved, type=ico, count=1
# ICO entry: w,h,colors,reserved,planes,bpp,size,offset
ico += struct.pack('<BBBBHHII', 64, 64, 0, 0, 1, 32, len(png_data), 22)
ico += png_data
with open('packaging/assets/icon.ico', 'wb') as f:
    f.write(ico)
print('icon.ico created')
"
```

- [ ] **Step 4: Copy PNG to ICNS placeholder**

```bash
cp packaging/assets/icon.png packaging/assets/icon.icns
```

- [ ] **Step 5: Commit**

```bash
git add packaging/assets/
git commit -m "feat: add packaging directory structure and placeholder icons"
```

---

### Task 2: Add frozen-detection and user-data-dir helpers

**Files:**
- Modify: `gui/gui/paths.py`

- [ ] **Step 1: Add helper functions to paths.py**

Read `gui/gui/paths.py` then insert the following code after the existing imports (after line 8 `from pathlib import Path`):

```python
import sys
import platform

def is_frozen() -> bool:
    """True when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False)

def get_app_root() -> Path:
    """Return the application root directory (works in dev and frozen modes)."""
    if is_frozen():
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent

def get_user_data_dir() -> Path:
    """Platform-standard user data directory for persistence."""
    app_name = "YoloLab"
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / app_name

def ensure_user_dirs() -> dict:
    """Create and return user data subdirectories."""
    data_dir = get_user_data_dir()
    dirs = {
        "models": data_dir / "models",
        "results": data_dir / "results",
        "logs": data_dir / "logs",
        "predict": data_dir / "predict",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

def get_preset_file() -> Path:
    """Return the path to presets.json (in user data dir when frozen)."""
    if is_frozen():
        return get_user_data_dir() / "presets.json"
    return Path(__file__).resolve().parent.parent / "gui" / "presets.json"
```

- [ ] **Step 2: Update the REPO_ROOT definition**

Change line 11 from:
```python
REPO_ROOT = Path(__file__).resolve().parent.parent
```
to:
```python
REPO_ROOT = get_app_root()
```

- [ ] **Step 3: Commit**

```bash
git add gui/gui/paths.py
git commit -m "feat: add frozen-detection and user-data-dir helpers"
```

---

### Task 3: Rework main.py entry point for multi-mode support

**Files:**
- Modify: `gui/gui/main.py`

- [ ] **Step 1: Replace the main() function and __main__ block**

Read `gui/gui/main.py` and replace the `main()` function (lines 1469-1481) with:

```python
def main():
    app = QApplication(sys.argv)
    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    font.setPixelSize(FONT_SIZE)
    app.setFont(font)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


def _run_engine_mode():
    """Entry point when launched as --engine-{train,infer,tool} by a subprocess."""
    from gui import train_engine, infer_engine

    mode = sys.argv[1]  # e.g. "--engine-train"
    # Remove the engine flag so argparse doesn't see it
    sys.argv.pop(1)

    if mode == "--engine-train":
        args = train_engine.parse_args()
        train_engine.run_non_interactive(args)
    elif mode == "--engine-infer":
        # infer_engine currently has inline argparse, import and run
        import gui.infer_engine as infer_engine
        infer_engine._loc = infer_engine._load_locale(
            next((sys.argv[i+1] for i, a in enumerate(sys.argv) if a == "--lang"), "zh")
        )
        # Re-parse since infer_engine uses __name__ == "__main__" pattern
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default=infer_engine.BEST_SEG_MODEL)
        parser.add_argument("--source", default=infer_engine.TEST_IMAGES_DIR)
        parser.add_argument("--save-dir", default=str(Path(infer_engine.PREDICT_DIR) / "predict_result"))
        parser.add_argument("--conf", type=float, default=0.406)
        parser.add_argument("--imgsz", type=int, default=640)
        parser.add_argument("--lang", default="zh")
        args = parser.parse_args(sys.argv[1:])
        infer_engine._loc = infer_engine._load_locale(args.lang)
        cfg = infer_engine.InferConfig(
            model_path=args.model, source=args.source, save_dir=args.save_dir,
            conf=args.conf, imgsz=args.imgsz,
            task_param_file=str(infer_engine._ENGINE_DIR / "infer_task_params.json"),
            out_suffix="_overlay.jpg",
        )
        inferencer = infer_engine.YOLOInferencer(cfg)
        inferencer.run()
    elif mode == "--engine-tool":
        # Tool args are passed as a JSON string (second argument)
        import json, importlib
        tool_args = json.loads(sys.argv[1])
        tool_idx = tool_args.pop("_tool_idx")

        TOOL_MODULES = [
            "tools.dataset_tools.create_empty_labels",
            "tools.dataset_tools.split_train_val.split_random_with_labels",
            "tools.dataset_tools.split_train_val_test.split_random_with_labels",
            "tools.dataset_tools.split_train_val.split_every_5th_with_labels",
            "tools.dataset_tools.split_images_only.split_random_images_only",
            "tools.dataset_tools.split_images_only.split_every_5th_images_only",
        ]
        mod = importlib.import_module(TOOL_MODULES[tool_idx])
        mod.run(**tool_args)
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("--engine-"):
        _run_engine_mode()
    else:
        main()
```

- [ ] **Step 2: Update PRESET_FILE definition**

Change line 17 from:
```python
PRESET_FILE = ROOT / "gui" / "presets.json"
```
to:
```python
from gui.paths import get_preset_file
PRESET_FILE = get_preset_file()
```

- [ ] **Step 3: Commit**

```bash
git add gui/gui/main.py
git commit -m "feat: add multi-mode entry point for frozen subprocess support"
```

---

### Task 4: Update workers.py cmd construction for frozen mode

**Files:**
- Modify: `gui/gui/workers.py`

- [ ] **Step 1: Update ROOT import and add frozen-aware helpers**

Read `gui/gui/workers.py` and replace line 13 (`ROOT = Path(__file__).resolve().parent.parent`) with:

```python
from gui.paths import is_frozen
ROOT = Path(__file__).resolve().parent.parent
```

- [ ] **Step 2: No other changes needed to workers.py**

The Worker classes (`TrainWorker`, `InferWorker`, `ToolWorker`) already accept a `cmd` list and run `subprocess.Popen(cmd, ...)`. The cmd construction changes happen in `main.py`.

- [ ] **Step 3: Update main.py cmd construction to use engine-mode flags**

Read `gui/gui/main.py` and update the training cmd construction (around line 1118-1138).

Replace the cmd block in `_on_start_train`:
```python
        cmd = [
            sys.executable, str(ROOT / "gui" / "train_engine.py"),
            "--lang", current_lang(),
            "--no-interactive",
            "--mode", str(mode),
            "--data-yaml", cfg.data_yaml,
            "--model-file", cfg.model_file,
            "--results-dir", cfg.results_dir,
            "--log-dir", cfg.log_dir,
            "--epochs", str(cfg.epochs),
            "--imgsz", str(cfg.imgsz),
            "--batch", str(cfg.batch),
            "--device", cfg.device,
            "--name", cfg.experiment_name,
        ]
        if use_aug:
            cmd.append("--use-augment")
        else:
            cmd.append("--no-augment")
        if mode == 3 and selected:
            cmd.extend(["--selected-exp", selected])
```

with:

```python
        cmd = [
            sys.executable, "--engine-train",
            "--lang", current_lang(),
            "--no-interactive",
            "--mode", str(mode),
            "--data-yaml", cfg.data_yaml,
            "--model-file", cfg.model_file,
            "--results-dir", cfg.results_dir,
            "--log-dir", cfg.log_dir,
            "--epochs", str(cfg.epochs),
            "--imgsz", str(cfg.imgsz),
            "--batch", str(cfg.batch),
            "--device", cfg.device,
            "--name", cfg.experiment_name,
        ]
        if use_aug:
            cmd.append("--use-augment")
        else:
            cmd.append("--no-augment")
        if mode == 3 and selected:
            cmd.extend(["--selected-exp", selected])
```

- [ ] **Step 4: Update inference cmd construction in _on_start_infer**

Replace the cmd block around line 1398-1406:
```python
        cmd = [
            sys.executable, str(ROOT / "gui" / "infer_engine.py"),
            "--lang", current_lang(),
            "--model", model_path,
            "--source", source,
            "--save-dir", save_dir,
            "--conf", str(conf_val),
            "--imgsz", str(imgsz_val),
        ]
```

with:

```python
        cmd = [
            sys.executable, "--engine-infer",
            "--lang", current_lang(),
            "--model", model_path,
            "--source", source,
            "--save-dir", save_dir,
            "--conf", str(conf_val),
            "--imgsz", str(imgsz_val),
        ]
```

- [ ] **Step 5: Update tool cmd construction in _on_run_tool**

Replace the entire cmd construction block (lines 498-521) in `_on_run_tool` with:

```python
        # Build tool args dict
        tool_args = {"_tool_idx": idx}
        if dataset_dir:
            tool_args["dataset_dir"] = dataset_dir
        spinners = self._tool_param_spinners[idx]
        if idx == 1:
            tool_args["val_ratio"] = spinners["val_ratio"].value() / 100.0
        elif idx == 2:
            tool_args["val_ratio"] = spinners["val_ratio"].value() / 100.0
            tool_args["test_ratio"] = spinners["test_ratio"].value() / 100.0
        elif idx == 3:
            tool_args["interval"] = int(spinners["interval"].value())
        elif idx == 4:
            tool_args["val_ratio"] = spinners["val_ratio"].value() / 100.0
        elif idx == 5:
            tool_args["interval"] = int(spinners["interval"].value())

        cmd = [sys.executable, "--engine-tool", json.dumps(tool_args, ensure_ascii=True)]
```

Note: Add `import json` to the top-level imports of main.py if not already present (it already is at line 7).

- [ ] **Step 6: Commit**

```bash
git add gui/gui/main.py
git commit -m "feat: use engine-mode flags in worker cmd construction for frozen compatibility"
```

---

### Task 5: Create PyInstaller spec file

**Files:**
- Create: `packaging/yolo_lab.spec`

- [ ] **Step 1: Write yolo_lab.spec**

```python
# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

a = Analysis(
    [str(PROJECT_ROOT / "gui" / "gui" / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[
        (str(PROJECT_ROOT / "gui" / "locales"), "locales"),
        (str(PROJECT_ROOT / "gui" / "gui" / "infer_task_params.json"), "gui"),
    ],
    hiddenimports=[
        "ultralytics",
        "ultralytics.nn",
        "ultralytics.nn.modules",
        "ultralytics.nn.tasks",
        "ultralytics.data",
        "ultralytics.data.build",
        "ultralytics.utils",
        "ultralytics.utils.checks",
        "ultralytics.utils.downloads",
        "ultralytics.engine",
        "ultralytics.engine.model",
        "ultralytics.engine.results",
        "torch",
        "cv2",
        "numpy",
        "yaml",
        "matplotlib",
        "matplotlib.backends.backend_agg",
        "PIL",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "tools.dataset_tools.create_empty_labels",
        "tools.dataset_tools.split_train_val.split_random_with_labels",
        "tools.dataset_tools.split_train_val_test.split_random_with_labels",
        "tools.dataset_tools.split_train_val.split_every_5th_with_labels",
        "tools.dataset_tools.split_images_only.split_random_images_only",
        "tools.dataset_tools.split_images_only.split_every_5th_images_only",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torchvision",
        "torchaudio",
        "PySide6.QtWebEngine",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebChannel",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtBluetooth",
        "PySide6.QtNfc",
        "PySide6.QtSensors",
        "PySide6.QtSerialPort",
        "PySide6.QtPositioning",
        "PySide6.QtLocation",
        "PySide6.QtRemoteObjects",
        "PySide6.QtTextToSpeech",
        "scipy",
        "pandas",
        "tcl",
        "tkinter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="YoloLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "packaging" / "assets" / "icon.ico")
        if sys.platform == "win32" else str(PROJECT_ROOT / "packaging" / "assets" / "icon.png"),
)
```

- [ ] **Step 2: Commit**

```bash
git add packaging/yolo_lab.spec
git commit -m "feat: add PyInstaller spec with volume optimizations"
```

---

### Task 6: Create Windows Inno Setup script

**Files:**
- Create: `packaging/windows/setup.iss`

- [ ] **Step 1: Write setup.iss**

```ini
#define AppName "YoloLab"
#define AppVersion GetEnv("APP_VERSION")
#define AppPublisher "YoloLab"
#define AppURL "https://github.com/user/yolo_lab_gui"
#define AppExeName "YoloLab.exe"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
LicenseFile=..\..\LICENSE
OutputDir=..\dist
OutputBaseFilename=YoloLab-{#AppVersion}-Setup
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64 arm64
ArchitecturesInstallIn64BitMode=x64 arm64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\YoloLab\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
```

- [ ] **Step 2: Commit**

```bash
git add packaging/windows/setup.iss
git commit -m "feat: add Windows Inno Setup installer script"
```

---

### Task 7: Create Linux AppImage recipe

**Files:**
- Create: `packaging/linux/build_appimage.sh`

- [ ] **Step 1: Write build_appimage.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

APP_NAME="YoloLab"
VERSION="${APP_VERSION:-0.1.0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST_DIR="$PROJECT_ROOT/packaging/dist"
APPDIR="$DIST_DIR/${APP_NAME}.AppDir"

# Clean previous
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/256x256/apps"

# Copy PyInstaller output
cp -r "$DIST_DIR/${APP_NAME}/"* "$APPDIR/usr/bin/"

# Create wrapper script
cat > "$APPDIR/AppRun" << 'WRAPPER'
#!/usr/bin/env bash
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/YoloLab" "$@"
WRAPPER
chmod +x "$APPDIR/AppRun"

# Create .desktop file
cat > "$APPDIR/usr/share/applications/yololab.desktop" << DESKTOP
[Desktop Entry]
Name=YoloLab
Comment=YOLO Segmentation Training & Inference GUI
Exec=AppRun
Icon=yololab
Terminal=false
Type=Application
Categories=Science;ArtificialIntelligence;
DESKTOP

# Copy icon
cp "$PROJECT_ROOT/packaging/assets/icon.png" \
   "$APPDIR/usr/share/icons/hicolor/256x256/apps/yololab.png"
cp "$PROJECT_ROOT/packaging/assets/icon.png" "$APPDIR/yololab.png"

# Download appimagetool if not present
APPIMAGETOOL="$DIST_DIR/appimagetool-x86_64.AppImage"
if [ ! -f "$APPIMAGETOOL" ]; then
    curl -sSL "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage" -o "$APPIMAGETOOL"
    chmod +x "$APPIMAGETOOL"
fi

# Build AppImage
ARCH=x86_64 "$APPIMAGETOOL" "$APPDIR" "$DIST_DIR/${APP_NAME}-${VERSION}-x86_64.AppImage"

echo "AppImage built: $DIST_DIR/${APP_NAME}-${VERSION}-x86_64.AppImage"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x packaging/linux/build_appimage.sh
git add packaging/linux/build_appimage.sh
git commit -m "feat: add Linux AppImage build script"
```

---

### Task 8: Create macOS DMG build script

**Files:**
- Create: `packaging/macos/build_dmg.sh`

- [ ] **Step 1: Write build_dmg.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

APP_NAME="YoloLab"
VERSION="${APP_VERSION:-0.1.0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST_DIR="$PROJECT_ROOT/packaging/dist"
APP_BUNDLE="$DIST_DIR/${APP_NAME}.app"

# Clean previous
rm -rf "$APP_BUNDLE"

# Create .app bundle structure
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"

# Copy PyInstaller output
cp -r "$DIST_DIR/${APP_NAME}/"* "$APP_BUNDLE/Contents/MacOS/"

# Create Info.plist
cat > "$APP_BUNDLE/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>com.yololab.app</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundleExecutable</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIconFile</key>
    <string>icon</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

# Copy icon
cp "$PROJECT_ROOT/packaging/assets/icon.icns" "$APP_BUNDLE/Contents/Resources/icon.icns"

# Create DMG
DMG_PATH="$DIST_DIR/${APP_NAME}-${VERSION}-$(uname -m).dmg"
rm -f "$DMG_PATH"
hdiutil create -volname "${APP_NAME}" -srcfolder "$APP_BUNDLE" -ov -format UDZO "$DMG_PATH"

echo "DMG built: $DMG_PATH"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x packaging/macos/build_dmg.sh
git add packaging/macos/build_dmg.sh
git commit -m "feat: add macOS DMG build script"
```

---

### Task 9: Create local build scripts

**Files:**
- Create: `packaging/build_local.sh`
- Create: `packaging/build_local.bat`

- [ ] **Step 1: Write build_local.sh (Linux/macOS)**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION="${VERSION:-0.1.0}"
CUDA="${CUDA:-0}"
PLATFORM="$(uname -s)"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --cuda) CUDA=1; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

export APP_VERSION="$VERSION"

echo "=== Building YoloLab v${VERSION} for ${PLATFORM} ==="

# Install dependencies
pip install --upgrade pip
if [ "$CUDA" = "1" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install ultralytics pyyaml pyinstaller
pip install PySide6

# Build with PyInstaller
cd "$PROJECT_ROOT"
pyinstaller --clean --noconfirm packaging/yolo_lab.spec

# Platform-specific packaging
if [ "$PLATFORM" = "Linux" ]; then
    bash "$SCRIPT_DIR/linux/build_appimage.sh"
elif [ "$PLATFORM" = "Darwin" ]; then
    bash "$SCRIPT_DIR/macos/build_dmg.sh"
fi

echo "=== Build complete: packaging/dist/ ==="
ls -lh "$SCRIPT_DIR/dist/"
```

- [ ] **Step 2: Write build_local.bat (Windows)**

```bat
@echo off
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set VERSION=0.1.0
set CUDA=0

:parse_args
if "%~1"=="" goto done_parsing
if "%~1"=="--version" (
    set VERSION=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--cuda" (
    set CUDA=1
    shift
    goto parse_args
)
echo Unknown: %~1
exit /b 1
:done_parsing

set APP_VERSION=%VERSION%

echo === Building YoloLab v%VERSION% for Windows ===

pip install --upgrade pip
if "%CUDA%"=="1" (
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    pip install torch --index-url https://download.pytorch.org/whl/cpu
)
pip install ultralytics pyyaml pyinstaller
pip install PySide6

cd /d "%PROJECT_ROOT%"
pyinstaller --clean --noconfirm packaging\yolo_lab.spec

echo === PyInstaller done, now run Inno Setup to create installer ===
echo === Open packaging\windows\setup.iss in Inno Setup Compiler ===
echo === Or install Inno Setup CLI and run: ===
echo ===   iscc packaging\windows\setup.iss ===

dir packaging\dist\
echo === Build complete ===
```

- [ ] **Step 3: Make executable and commit**

```bash
chmod +x packaging/build_local.sh
git add packaging/build_local.sh packaging/build_local.bat
git commit -m "feat: add local build scripts for Linux, macOS, and Windows"
```

---

### Task 10: Create GitHub Actions release workflow

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Write release.yml**

```yaml
name: Build & Release

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
    inputs:
      cuda:
        description: 'Build CUDA version'
        type: boolean
        default: false

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies (CPU)
        if: ${{ !inputs.cuda }}
        run: |
          pip install torch --index-url https://download.pytorch.org/whl/cpu
          pip install ultralytics pyyaml pyinstaller PySide6

      - name: Install dependencies (CUDA)
        if: ${{ inputs.cuda }}
        run: |
          pip install torch --index-url https://download.pytorch.org/whl/cu121
          pip install ultralytics pyyaml pyinstaller PySide6

      - name: Build with PyInstaller
        run: pyinstaller --noconfirm packaging/yolo_lab.spec
        env:
          APP_VERSION: ${{ github.ref_name }}

      - name: Install Inno Setup
        uses: nebularnoah/setup-iscc@v1

      - name: Create installer
        run: iscc packaging/windows/setup.iss
        env:
          APP_VERSION: ${{ github.ref_name }}

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: YoloLab-${{ github.ref_name }}-Setup.exe
          path: packaging/dist/YoloLab-${{ github.ref_name }}-Setup.exe

  build-linux:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install system deps
        run: |
          sudo apt-get update
          sudo apt-get install -y libgl1-mesa-glx libegl1 libxcb-icccm4 \
            libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0 \
            libxcb-shape0 libxcb-cursor0 libfuse2

      - name: Install Python dependencies (CPU)
        run: |
          pip install torch --index-url https://download.pytorch.org/whl/cpu
          pip install ultralytics pyyaml pyinstaller PySide6

      - name: Build with PyInstaller
        run: pyinstaller --noconfirm packaging/yolo_lab.spec
        env:
          APP_VERSION: ${{ github.ref_name }}

      - name: Build AppImage
        run: bash packaging/linux/build_appimage.sh
        env:
          APP_VERSION: ${{ github.ref_name }}

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: YoloLab-${{ github.ref_name }}-x86_64.AppImage
          path: packaging/dist/YoloLab-${{ github.ref_name }}-x86_64.AppImage

  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies (CPU)
        run: |
          pip install torch --index-url https://download.pytorch.org/whl/cpu
          pip install ultralytics pyyaml pyinstaller PySide6

      - name: Build with PyInstaller
        run: pyinstaller --noconfirm packaging/yolo_lab.spec
        env:
          APP_VERSION: ${{ github.ref_name }}

      - name: Build DMG
        run: bash packaging/macos/build_dmg.sh
        env:
          APP_VERSION: ${{ github.ref_name }}

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: YoloLab-${{ github.ref_name }}-${{ runner.arch }}.dmg
          path: packaging/dist/YoloLab-${{ github.ref_name }}-*.dmg

  release:
    needs: [build-windows, build-linux, build-macos]
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          path: artifacts

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          name: ${{ github.ref_name }}
          files: artifacts/**/*
          generate_release_notes: true
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "feat: add GitHub Actions release workflow for all platforms"
```

---

### Task 11: Verify local build

- [ ] **Step 1: Run PyInstaller build locally**

```bash
cd /home/ljz/vibe_coding/yolo
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics pyyaml pyinstaller PySide6
pyinstaller --noconfirm packaging/yolo_lab.spec
```

- [ ] **Step 2: Verify the built exe exists and launches**

```bash
ls -lh packaging/dist/YoloLab/
file packaging/dist/YoloLab/YoloLab
```

- [ ] **Step 3: Test launch (headless check for import errors)**

```bash
# Quick smoke test: verify no import errors
timeout 5 ./packaging/dist/YoloLab/YoloLab 2>&1 || true
```

- [ ] **Step 4: Fix any missing imports or data files**

If the exe fails to launch, use this to debug:
```bash
./packaging/dist/YoloLab/YoloLab --help 2>&1 | head -50
```

Common issues and fixes:
- Missing ultralytics submodule: add to `hiddenimports` in spec
- Missing locale files: verify `datas` paths in spec
- Missing Qt platform plugin: add `--add-data` for PySide6 plugins

- [ ] **Step 5: Commit any fixes**

```bash
git add -A && git commit -m "fix: PyInstaller spec adjustments from local build test"
```

---

### Post-Implementation: How to Release

```bash
# 1. Tag a version
git tag v0.1.0

# 2. Push tag → CI auto-builds
git push origin v0.1.0

# 3. Go to GitHub Releases page → download installers

# 4. Or build locally:
./packaging/build_local.sh --version 0.1.0        # Linux/macOS
.\packaging\build_local.bat --version 0.1.0        # Windows
```
