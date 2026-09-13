#!/bin/bash
#SBATCH --job-name=SWEEP_ALL_CKPTS_CPU
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/SWEEP_ALL_CKPTS_CPU_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/SWEEP_ALL_CKPTS_CPU_%j.err

# ==============================================================================
# QUY TRÌNH:
# 1. Quét qua TẤT CẢ các checkpoints trong outputs/papers/rafdb_v1_clean_evolution/checkpoints/
#    (Gồm các checkpoint trong best/, best_loss/, periodic/, last/)
# 2. Với MỖI checkpoint:
#    - Trích xuất Logits ảnh gốc + ảnh lật trên tập VALIDATION.
#    - Sweep tìm bộ trọng số TTA (w_orig, w_flip) tối ưu nhất trên VALIDATION.
#    - Áp dụng bộ trọng số TTA tối ưu từ Val để đánh giá trên tập TEST (Zero Leakage).
# 3. Tổng hợp bảng so sánh, tự động chọn ra Checkpoint Vô địch dựa trên Validation!
# 4. Tìm kiếm tổ hợp Ensemble tối ưu từ các checkpoints tốt nhất.
# 5. Chạy thuần túy trên CPU (0 MB VRAM, CUDA_VISIBLE_DEVICES="-1").
# ==============================================================================

set -euo pipefail

ROOT="/home/ptbao/projects/FER2013_MGR_CNN"
cd "$ROOT"
mkdir -p logs

# Ép chặt CPU mode
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
echo " SBATCH: SWEEP TTA TỪNG CHECKPOINT TRÊN VAL -> CHỌN TEST (CPU)"
echo " Date       : $(date)"
echo " Host       : $(hostname)"
echo " Config     : $CONFIG"
echo " Exp Dir    : $EXP_DIR"
echo " Cores      : $NCPUS threads"
echo " Mode       : PURE CPU (CUDA_VISIBLE_DEVICES=-1, GPU Disabled)"
echo " Job ID     : ${SLURM_JOB_ID:-standalone}"
echo "=========================================================================="

# Chạy script tối ưu TTA & Ensemble toàn diện trên CPU
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

echo ""
echo "=========================================================================="
echo " [SUCCESS] TTA Sweep & Checkpoint Selection Completed at $(date)"
echo " Results saved in: $EXP_DIR"
echo "=========================================================================="
