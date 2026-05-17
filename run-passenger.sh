#!/bin/bash
# Launch Passenger image comparison tool for a batch output directory.
#
# Usage:
#   ./run-passenger.sh                                          # tuning2 (default)
#   ./run-passenger.sh ~/claude/comfyui/output/quality-comparison
#   ./run-passenger.sh ~/claude/comfyui/output/tuning2
#
# Passenger serves at http://localhost:8189 in tournament mode:
# pick the winner of each head-to-head matchup until one image per subject remains.

set -euo pipefail

DIR="${1:-$HOME/claude/comfyui/output/tuning2}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$DIR" ]; then
  echo "ERROR: Directory not found: $DIR" >&2
  exit 1
fi

if [ ! -f "$DIR/manifest.json" ]; then
  echo "ERROR: No manifest.json found in $DIR" >&2
  echo "Run a batch script first:" >&2
  echo "  python3 run-tuning-batch.py" >&2
  echo "  python3 run-quality-comparison.py" >&2
  exit 1
fi

# Kill any existing passenger instance on port 8189
pkill -f "passenger\.py" 2>/dev/null || true
sleep 0.3

python3 "$SCRIPT_DIR/passenger.py" "$DIR" &
PID=$!
sleep 0.5

if kill -0 "$PID" 2>/dev/null; then
  echo "Passenger running (PID $PID)"
  echo "Open: http://localhost:8189"
  echo "Dir:  $DIR"
else
  echo "ERROR: Passenger failed to start" >&2
  exit 1
fi
