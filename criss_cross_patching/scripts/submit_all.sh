#!/bin/bash
# Submit the full patching sweep: 4 models × 2 patching kinds = 8 jobs,
# each covering all 4 directions in one process.

set -eu
HERE=$(cd "$(dirname "$0")" && pwd)

for model in qwen2vl qwen3vl gemma3 pixtral; do
  for kind in object strip; do
    "$HERE/run_all_directions.sh" "$model" "$kind"
  done
done
