#!/bin/bash
set -euo pipefail

INSTALL_DIR="/opt/YoloLab"
BIN_LINK="/usr/local/bin/yolo-lab"
ICON_DIR="/usr/share/icons/hicolor/256x256/apps"
APP_DIR="/usr/share/applications"

if [ "$(id -u)" -ne 0 ]; then
    echo "请使用 sudo 运行: sudo ./install.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APPIMAGE=$(ls "$SCRIPT_DIR/dist"/YoloLab*.AppImage 2>/dev/null | head -1)

if [ -z "$APPIMAGE" ]; then
    echo "未找到 AppImage，请先运行 build.sh"
    exit 1
fi

echo "==> 安装 YOLO-LAB..."

# Install app
mkdir -p "$INSTALL_DIR"
cp "$APPIMAGE" "$INSTALL_DIR/YoloLab.AppImage"
chmod +x "$INSTALL_DIR/YoloLab.AppImage"

# Install icon
mkdir -p "$ICON_DIR"
cp "$SCRIPT_DIR/assets/icon.png" "$ICON_DIR/yolo-lab.png"

# Create symlink
ln -sf "$INSTALL_DIR/YoloLab.AppImage" "$BIN_LINK"

# Install desktop entry
mkdir -p "$APP_DIR"
cp "$SCRIPT_DIR/yolo_lab.desktop" "$APP_DIR/"

echo "✓ 安装完成！"
echo "  GUI: 从应用菜单启动 或 终端输入 yolo-lab"
echo "  CLI: yolo-lab train --epochs 100"
