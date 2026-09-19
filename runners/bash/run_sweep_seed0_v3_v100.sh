#!/bin/bash
# ==============================================================================
# Script chạy trực tiếp trên GPU V100 (Interactive):
# Quét từng checkpoint -> Sweep TTA trên Val -> Test -> Ensemble
# ==============================================================================

set -euo pipefail

ROOT="/home/ptbao/projects/FER2013_MGR_CNN"
cd "$ROOT"
mkdir -p "$ROOT/logs"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_convnext_base_ms1m_adaptive_siglip2_confusion_seed0.yaml"
EXP_DIR="$ROOT/outputs/papers/siglip2-confusion-seed0_v3"

SITE_PACKAGES=$("$FER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB="$SITE_PACKAGES/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "=========================================================================="
echo " CHẠY TRỰC TIẾP TRÊN GPU V100: TTA SWEEP & ENSEMBLE SEED 0 V3"
echo " Exp Dir : $EXP_DIR"
echo "=========================================================================="

nvidia-smi

"$FER_PY" -u scripts/sweep_tta_and_ensemble_all_checkpoints.py \
    --config "$CONFIG" \
    --exp-dir "$EXP_DIR" \
    --step 0.05 \
    --include-best-loss \
    --include-best-macro-f1 \
    --num-comb-samples 50000 \
    --save-individual \
    --output "$EXP_DIR/val_tuned_tta_ensemble_report.json"
