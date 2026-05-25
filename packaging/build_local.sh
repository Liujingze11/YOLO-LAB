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
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics pyyaml pyinstaller PySide6

# Build with PyInstaller (always CPU for default package)
cd "$PROJECT_ROOT"
pyinstaller --clean --noconfirm packaging/yolo_lab.spec

# GPU bundle (optional, built separately when --cuda flag is set)
if [ "$CUDA" = "1" ]; then
    echo "=== Building GPU bundle ==="
    GPU_DIR="$PROJECT_ROOT/dist/gpu_bundle"
    rm -rf "$GPU_DIR"
    mkdir -p "$GPU_DIR"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --target "$GPU_DIR"
    cd "$PROJECT_ROOT/dist"
    zip -r "gpu_bundle.zip" "gpu_bundle/"
    rm -rf "gpu_bundle/"
    echo "GPU bundle: $PROJECT_ROOT/dist/gpu_bundle.zip"
fi

# Platform-specific packaging
if [ "$PLATFORM" = "Linux" ]; then
    bash "$SCRIPT_DIR/linux/build_appimage.sh"
elif [ "$PLATFORM" = "Darwin" ]; then
    bash "$SCRIPT_DIR/macos/build_dmg.sh"
fi

echo "=== Build complete: dist/ ==="
ls -lh "$PROJECT_ROOT/dist/"
