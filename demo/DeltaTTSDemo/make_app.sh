#!/usr/bin/env bash
# Build DeltaTTSDemo.app: the release binary in an app bundle whose Info.plist declares the
# microphone use (macOS asks for permission only for bundled apps), signed ad hoc.
#
#   demo/DeltaTTSDemo/make_app.sh [output_dir]     (default: demo/DeltaTTSDemo/build)
#   open demo/DeltaTTSDemo/build/DeltaTTSDemo.app
#
# JOBS sets the build parallelism (default 2).
set -euo pipefail

cd "$(dirname "$0")"
OUT="${1:-build}"
APP="$OUT/DeltaTTSDemo.app"
BUNDLE_ID="io.github.slpr-lab.DeltaTTSDemo"

swift build -c release -j "${JOBS:-2}"
BIN="$(swift build -c release --show-bin-path)/DeltaTTSDemo"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/DeltaTTSDemo"

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>DeltaTTSDemo</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>DeltaTTSDemo</string>
    <key>CFBundleDisplayName</key>
    <string>DELTA-TTS Demo</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>26.0</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.utilities</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>DELTA-TTS records one sentence of your voice to speak text in it. The recording stays on this Mac and is deleted when you press Next visitor or quit.</string>
</dict>
</plist>
EOF
plutil -lint "$APP/Contents/Info.plist" >/dev/null

codesign --force --sign - --timestamp=none "$APP"
codesign --verify --strict "$APP"
echo "Built $(cd "$OUT" && pwd)/DeltaTTSDemo.app"
