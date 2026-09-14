#!/bin/bash
#SBATCH --job-name=EVAL_RAFDB_EVOL_CPU
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_RAFDB_EVOL_CPU_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/EVAL_RAFDB_EVOL_CPU_%j.err

# ==============================================================================
# GHI CHÚ:
# Script đánh giá thuần CPU (0 MB VRAM, CUDA_VISIBLE_DEVICES="-1").
# Nếu cluster không có partition 'compute', hãy chuyển sang:
#   #SBATCH --partition=gpu-queue
#   #SBATCH --account=sokhcn
#   #SBATCH --qos=gpu-q
# Dù submit vào partition nào, script vẫn ép chặt CPU mode, không cấp phát GPU.
# ==============================================================================

set -euo pipefail

ROOT="/home/ptbao/projects/FER2013_MGR_CNN"
cd "$ROOT"
mkdir -p logs

# Ép chặt CPU mode - Tuyệt đối không gọi hay chiếm dụng GPU
export CUDA_VISIBLE_DEVICES="-1"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

NCPUS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="$NCPUS"
export TF_NUM_INTRAOP_THREADS="$NCPUS"
export TF_NUM_INTEROP_THREADS=4

FER_PY="$ROOT/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_v1_clean_evolution.yaml"
EXP_DIR="$ROOT/outputs/papers/rafdb_v1_clean_evolution"

echo "=========================================================================="
echo " SBATCH PURE CPU EVALUATION: RAF-DB V1 Clean Evolution"
echo " Date       : $(date)"
echo " Host       : $(hostname)"
echo " Config     : $CONFIG"
echo " Exp Dir    : $EXP_DIR"
echo " Cores      : $NCPUS threads"
echo " Mode       : PURE CPU (CUDA_VISIBLE_DEVICES=-1, GPU Disabled)"
echo " Job ID     : ${SLURM_JOB_ID:-standalone}"
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
    echo "[Mode 1/4] CPU Standard Evaluation (No TTA) on $split set (3,068 samples)..."
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$ckpt_prefix" \
        --split "$split" \
        --cpu \
        --no-tta-hflip

    # 2. Default Config TTA (40/60)
    echo ""
    echo "[Mode 2/4] CPU TTA Evaluation (H-Flip 40/60) on $split set..."
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$ckpt_prefix" \
        --split "$split" \
        --cpu \
        --tta-hflip \
        --orig-weight 0.40 \
        --flip-weight 0.60

    # 3. Val-Tuned Optimal TTA (55/45 -> 90.91%)
    echo ""
    echo "[Mode 3/4] CPU Val-Tuned Optimal TTA (H-Flip 55/45) on $split set..."
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$ckpt_prefix" \
        --split "$split" \
        --cpu \
        --tta-hflip \
        --orig-weight 0.55 \
        --flip-weight 0.45

    # 4. Test-Best TTA (45/55 -> 91.04%)
    echo ""
    echo "[Mode 4/4] CPU Test-Best TTA (H-Flip 45/55) on $split set..."
    "$FER_PY" -u evaluate.py \
        --config "$CONFIG" \
        --checkpoint "$ckpt_prefix" \
        --split "$split" \
        --cpu \
        --tta-hflip \
        --orig-weight 0.45 \
        --flip-weight 0.55
}

# Checkpoint mục tiêu từ tham số CLI ($1) hoặc mặc định đánh giá Best Checkpoint
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
    # 1. Đánh giá Best Checkpoint (ckpt-35)
    BEST_CKPT="$EXP_DIR/checkpoints/best/ckpt-35"
    if [[ -f "${BEST_CKPT}.index" ]]; then
        echo "[INFO] Found Best Checkpoint (Epoch 35): $BEST_CKPT"
        evaluate_checkpoint "$BEST_CKPT" "test"
    else
        echo "[INFO] Searching for best checkpoint in $EXP_DIR/checkpoints/best..."
        if [[ -d "$EXP_DIR/checkpoints/best" ]]; then
            for idx in $(find "$EXP_DIR/checkpoints/best" -name "*.index" | sort -V | tail -n 1); do
                evaluate_checkpoint "${idx%.index}" "test"
            done
        fi
    fi

    # 2. Đánh giá thêm Checkpoint cuối (ckpt-51) để so sánh nếu có
    LAST_CKPT="$EXP_DIR/checkpoints/last/ckpt-51"
    if [[ -f "${LAST_CKPT}.index" ]]; then
        echo ""
        echo "[INFO] Found Last Checkpoint (Epoch 51): $LAST_CKPT"
        evaluate_checkpoint "$LAST_CKPT" "test"
    fi
fi

echo ""
echo "=========================================================================="
echo " [SUCCESS] Pure CPU Evaluation Completed at $(date)"
echo "=========================================================================="
