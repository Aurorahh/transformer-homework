#!/bin/bash
# 文件: scripts/run_sensitivity.sh

set -e
SEED=42
echo "Running Sensitivity Analysis for Number of Heads (Seed: $SEED)"

# 变体 1: 4 Heads
echo "Running with N_HEADS = 4"
python train.py --experiment-name "heads_4_seed$SEED" --seed $SEED --n-layers 3 --n-heads 4

# 变体 2: 16 Heads 
echo "Running with N_HEADS = 16"
python train.py --experiment-name "heads_16_seed$SEED" --seed $SEED --n-layers 3 --n-heads 16

echo "Sensitivity analysis complete. Check results/tables/experiment_summary.md for comparison."