#!/bin/bash
#SBATCH --job-name=SWEEP_SEED0_V3_CPU
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/SWEEP_SEED0_V3_CPU_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/SWEEP_SEED0_V3_CPU_%j.err

# ==============================================================================
# Script chạy sweep TTA & Ensemble thuần CPU cho thư mục siglip2-confusion-seed0_v3
# Không chạm vào GPU, an toàn 100%.
# ==============================================================================

set -euo pipefail

ROOT="/home/ptbao/projects/FER2013_MGR_CNN"
cd "$ROOT"
mkdir -p logs

export CUDA_VISIBLE_DEVICES="-1"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

NCPUS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="$NCPUS"
export TF_NUM_INTRAOP_THREADS="$NCPUS"
export TF_NUM_INTEROP_THREADS=4

FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_convnext_base_ms1m_adaptive_siglip2_confusion_seed0.yaml"
EXP_DIR="$ROOT/outputs/papers/siglip2-confusion-seed0_v3"

echo "=========================================================================="
echo " Quét TTA & Ensemble trên toàn bộ Checkpoint của siglip2-confusion-seed0_v3"
echo " Cores   : $NCPUS threads (CPU Mode, 0 MB VRAM)"
echo " Exp Dir : $EXP_DIR"
echo "=========================================================================="

"$FER_PY" -u scripts/sweep_tta_and_ensemble_all_checkpoints.py \
    --config "$CONFIG" \
    --exp-dir "$EXP_DIR" \
    --step 0.05 \
    --include-best-loss \
    --include-best-macro-f1 \
    --num-comb-samples 20000 \
    --save-individual \
    --cpu

echo "=========================================================================="
echo " Hoàn thành lúc $(date)"
echo "=========================================================================="
