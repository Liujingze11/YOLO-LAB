#!/bin/bash
set -euo pipefail

VERSION="${1:-0.2.0}"
LINUX_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="$LINUX_DIR/dist"
SPEC="$LINUX_DIR/yolo_lab.spec"

echo "==> Building YoloLab v${VERSION} for Linux..."

# Clean
rm -rf "$DIST_DIR" "$LINUX_DIR/build"

# PyInstaller
pyinstaller --distpath "$DIST_DIR" --workpath "$LINUX_DIR/build" "$SPEC"

echo "==> Build complete: $DIST_DIR/YoloLab"
