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
    pip install torch --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install ultralytics pyyaml pyinstaller PySide6

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
