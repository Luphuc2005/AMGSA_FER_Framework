#!/bin/bash
# ==============================================================================
# Script chạy trực tiếp (Interactive/Bash) trên CPU:
# Quét từng checkpoint -> Sweep TTA trên Val -> Chọn checkpoint & trọng số -> Test
# An toàn 100%: CUDA_VISIBLE_DEVICES="-1"
# ==============================================================================

set -euo pipefail

ROOT="/home/ptbao/projects/FER2013_MGR_CNN"
cd "$ROOT"
mkdir -p logs

export CUDA_VISIBLE_DEVICES="-1"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

NCPUS="${NCPUS:-16}"
export OMP_NUM_THREADS="$NCPUS"
export TF_NUM_INTRAOP_THREADS="$NCPUS"
export TF_NUM_INTEROP_THREADS=4

FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_v1_clean_evolution.yaml"
EXP_DIR="$ROOT/outputs/papers/rafdb_v1_clean_evolution"

echo "=========================================================================="
echo " DIRECT CPU: SWEEP TTA TỪNG CHECKPOINT TRÊN VAL -> TEST"
echo " Config     : $CONFIG"
echo " Exp Dir    : $EXP_DIR"
echo " Cores      : $NCPUS threads"
echo " Mode       : PURE CPU (CUDA_VISIBLE_DEVICES=-1, GPU Disabled)"
echo "=========================================================================="

"$FER_PY" -u scripts/sweep_tta_and_ensemble_all_checkpoints.py \
    --config "$CONFIG" \
    --exp-dir "$EXP_DIR" \
    --step 0.05 \
    --include-best-loss \
    --include-best-macro-f1 \
    --include-periodic \
    --num-comb-samples 20000 \
    --save-individual \
    --cpu
