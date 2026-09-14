#!/bin/bash
# ==============================================================================
# Script Evaluation thuần CPU cho RAF-DB V1 Clean Evolution (Job 6787)
# An toàn 100%: CUDA_VISIBLE_DEVICES="-1", không chiếm GPU, không đụng VRAM.
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
echo " Direct CPU Evaluation: RAF-DB V1 Clean Evolution"
echo " Config  : $CONFIG"
echo " Exp Dir : $EXP_DIR"
echo " Cores   : $NCPUS threads"
echo " Mode    : PURE CPU (CUDA_VISIBLE_DEVICES=-1, GPU Disabled)"
echo "=========================================================================="

evaluate_checkpoint() {
    local ckpt_prefix="$1"
    local split="${2:-test}"
    echo ""
    echo "=========================================================================="
    echo ">>> EVALUATING CHECKPOINT (CPU) on Split [$split]: $ckpt_prefix"
    echo "=========================================================================="

    # 1. No-TTA Evaluation
    echo ""
    echo "[1/3] Standard Evaluation (No TTA) on $split set..."
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$ckpt_prefix" \
        --split "$split" \
        --cpu \
        --no-tta-hflip

    # 2. Default Config TTA (40/60)
    echo ""
    echo "[2/3] Config TTA Evaluation (H-Flip 40/60) on $split set..."
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$ckpt_prefix" \
        --split "$split" \
        --cpu \
        --tta-hflip \
        --orig-weight 0.40 \
        --flip-weight 0.60

    # 3. Optimal Val-Tuned TTA (55/45)
    echo ""
    echo "[3/3] Val-Tuned Optimal TTA (H-Flip 55/45) on $split set..."
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$ckpt_prefix" \
        --split "$split" \
        --cpu \
        --tta-hflip \
        --orig-weight 0.55 \
        --flip-weight 0.45
}

TARGET_CKPT="${1:-}"

if [[ -n "$TARGET_CKPT" ]]; then
    if [[ "$TARGET_CKPT" == *.index ]]; then
        TARGET_CKPT="${TARGET_CKPT%.index}"
    fi
    if [[ ! -f "${TARGET_CKPT}.index" ]]; then
        if [[ -f "$EXP_DIR/checkpoints/best/${TARGET_CKPT}.index" ]]; then
            TARGET_CKPT="$EXP_DIR/checkpoints/best/${TARGET_CKPT}"
        elif [[ -f "$EXP_DIR/checkpoints/last/${TARGET_CKPT}.index" ]]; then
            TARGET_CKPT="$EXP_DIR/checkpoints/last/${TARGET_CKPT}"
        elif [[ -f "$EXP_DIR/checkpoints/periodic/${TARGET_CKPT}.index" ]]; then
            TARGET_CKPT="$EXP_DIR/checkpoints/periodic/${TARGET_CKPT}"
        elif [[ -f "$ROOT/${TARGET_CKPT}.index" ]]; then
            TARGET_CKPT="$ROOT/${TARGET_CKPT}"
        else
            echo "[ERROR] Checkpoint not found: ${TARGET_CKPT}.index"
            exit 1
        fi
    fi
    evaluate_checkpoint "$TARGET_CKPT" "test"
else
    # Mặc định tìm và evaluate Best Checkpoint (ckpt-35)
    BEST_CKPT="$EXP_DIR/checkpoints/best/ckpt-35"
    if [[ -f "${BEST_CKPT}.index" ]]; then
        echo "[INFO] Found Best Checkpoint (Epoch 35): $BEST_CKPT"
        evaluate_checkpoint "$BEST_CKPT" "test"
    else
        echo "[INFO] Searching for available checkpoints in $EXP_DIR/checkpoints..."
        FOUND=0
        for dir in "best" "last" "periodic"; do
            if [[ -d "$EXP_DIR/checkpoints/$dir" ]]; then
                for idx in $(find "$EXP_DIR/checkpoints/$dir" -name "*.index" | sort -V); do
                    prefix="${idx%.index}"
                    evaluate_checkpoint "$prefix" "test"
                    FOUND=$((FOUND + 1))
                done
            fi
        done
        if [[ $FOUND -eq 0 ]]; then
            echo "[WARNING] No checkpoints found in $EXP_DIR"
        fi
    fi
fi

echo ""
echo "=========================================================================="
echo " [SUCCESS] CPU Evaluation Completed at $(date)"
echo "=========================================================================="
