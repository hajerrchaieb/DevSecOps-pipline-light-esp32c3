#!/bin/bash
# AutoFix — apply all generated patches to ESP-Matter source
# Generated: 2026-04-25 10:32
#
# Called by ci.yml Stage 4b BEFORE the second build:
#   - name: 🩹 Apply AutoFix patches
#     run: bash reports/patches/APPLY_ALL.sh
#
set -e

# Resolve ESP-Matter source dir (same logic as agents)
ESP_MAIN=''
for CANDIDATE in \
    'esp-matter/examples/light/main' \
    '../esp-matter/examples/light/main' \
    '/opt/espressif/esp-matter/examples/light/main'; do
  if [ -d "$CANDIDATE" ]; then ESP_MAIN="$CANDIDATE"; break; fi
done

if [ -z "$ESP_MAIN" ]; then
  echo '[AutoFix] ERROR: ESP-Matter source directory not found'
  exit 1
fi
echo "[AutoFix] Applying patches to: $ESP_MAIN"

PATCHES_DIR="$(dirname "$0")"
APPLIED=0
SKIPPED=0

echo '[AutoFix] → fix_app_main.cpp.patch'
if git apply --check --directory="$ESP_MAIN" "/home/runner/work/DevSecOps-pipline-light-esp32c3/DevSecOps-pipline-light-esp32c3/reports/patches/fix_app_main.cpp.patch" 2>/dev/null; then
  git apply --directory="$ESP_MAIN" "/home/runner/work/DevSecOps-pipline-light-esp32c3/DevSecOps-pipline-light-esp32c3/reports/patches/fix_app_main.cpp.patch"
  APPLIED=$((APPLIED+1))
  echo '  ✅ Applied'
else
  echo '  ⚠️  Skipped (already applied or conflicts)'
  SKIPPED=$((SKIPPED+1))
fi

echo '[AutoFix] → fix_tools_qemu_fault_runner.py.patch'
if git apply --check --directory="$ESP_MAIN" "/home/runner/work/DevSecOps-pipline-light-esp32c3/DevSecOps-pipline-light-esp32c3/reports/patches/fix_tools_qemu_fault_runner.py.patch" 2>/dev/null; then
  git apply --directory="$ESP_MAIN" "/home/runner/work/DevSecOps-pipline-light-esp32c3/DevSecOps-pipline-light-esp32c3/reports/patches/fix_tools_qemu_fault_runner.py.patch"
  APPLIED=$((APPLIED+1))
  echo '  ✅ Applied'
else
  echo '  ⚠️  Skipped (already applied or conflicts)'
  SKIPPED=$((SKIPPED+1))
fi

echo ""
echo "[AutoFix] Done — Applied: $APPLIED | Skipped: $SKIPPED"

# Verify build still compiles after patches
if command -v idf.py &>/dev/null; then
  echo '[AutoFix] Verifying build...'
  cd "$ESP_MAIN"
  idf.py build 2>&1 | tail -8
fi