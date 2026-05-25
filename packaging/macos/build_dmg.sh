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
