#!/usr/bin/env bash
set -euo pipefail
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-/ssd/sy/kailang/clean3/curriculum_20260915}"
case "$TARGET" in /ssd/sy/kailang/*) ;; *) echo "target outside kailang" >&2; exit 2;; esac
test -f "$TARGET/assets/retarget/object_0_textured.usd"
test -f "$TARGET/assets/retarget/object_1_textured.usd"
test -d /ssd/sy/kailang/pour17/direct58d
cd "$BUNDLE" && sha256sum -c FILES.sha256
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP="$TARGET/backups/clean3_before_current_20260916_$STAMP"
mkdir -p "$BACKUP/data" "$TARGET/tasks/h2s2r_clean3" "$TARGET/data"
cp -a "$TARGET/tasks/h2s2r_clean3/." "$BACKUP/" 2>/dev/null || true
for name in clean3_c3p1_runtime_inputs.npz clean3_residual_controller_inputs.npz; do
  if test -f "$TARGET/data/$name"; then cp -a "$TARGET/data/$name" "$BACKUP/data/$name"; fi
done
cp -a "$BUNDLE/overlay/tasks/h2s2r_clean3/." "$TARGET/tasks/h2s2r_clean3/"
cp -a "$BUNDLE/overlay/data/." "$TARGET/data/"
test ! -e "$TARGET/ours_stage1_overlay.new"
cp -a "$BUNDLE/overlay/ours_stage1_overlay" "$TARGET/ours_stage1_overlay.new"
if test -e "$TARGET/ours_stage1_overlay"; then mv "$TARGET/ours_stage1_overlay" "$BACKUP/ours_stage1_overlay"; fi
mv "$TARGET/ours_stage1_overlay.new" "$TARGET/ours_stage1_overlay"
echo "deployed: $TARGET"
echo "backup: $BACKUP"
