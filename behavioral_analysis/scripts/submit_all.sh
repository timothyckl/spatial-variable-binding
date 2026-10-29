#!/bin/bash
# Submit all 16 behavioral cells (4 models × 4 tasks) in parallel.
# Each cell is one SLURM job; jobs run independently.

set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

for model in qwen2vl qwen3vl gemma3 pixtral; do
  for task in squares shapes objects whatsup; do
    "$HERE/run_cell.sh" "$model" "$task"
  done
done
