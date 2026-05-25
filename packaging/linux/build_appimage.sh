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
