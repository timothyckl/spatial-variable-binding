#!/bin/bash
# Submit the section-5.2.1 probing sweep: 4 models × 3 synthetic settings × 2
# orientations = 24 jobs, plus 4 WhatsUp jobs (horizontal-only, one per model).
# Skip the grid extension by default — add it explicitly if you want App C.6.

set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

for model in qwen2vl qwen3vl gemma3 pixtral; do
  for setting in squares shapes objects; do
    for orientation in horizontal vertical; do
      "$HERE/run_cell.sh" "$model" "$setting" "$orientation"
    done
  done
  "$HERE/run_cell.sh" "$model" whatsup horizontal
done
