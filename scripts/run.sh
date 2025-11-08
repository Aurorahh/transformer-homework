#!/bin/bash
# 文件: scripts/run.sh

set -e

# 固定的随机种子确保可复现性
SEED=42

echo "Running Baseline Machine Translation Experiment (Seed: $SEED)..."
# 使用默认的 n_layers=3, n_heads=8
python train.py --experiment-name "baseline_mt_seed$SEED" --seed $SEED

echo "-------------------------------------"
echo "Running Ablation Study (No Positional Encoding, Seed: $SEED)..."
python train.py --experiment-name "no_pos_encoding_mt_seed$SEED" --no-pos-encoding --seed $SEED

echo "All main experiments finished."
echo "Check 'results/tables/experiment_summary.md' for results."