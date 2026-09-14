#!/bin/bash
#SBATCH --job-name=SWEEP_SEED0_V3_GPU
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/SWEEP_SEED0_V3_GPU_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/SWEEP_SEED0_V3_GPU_%j.err

# ==============================================================================
# QUY TRÌNH CHẠY TRÊN GPU V100 (TỐC ĐỘ CỰC NHANH - VÀI GIÂY / CHECKPOINT):
# 1. Quét qua TẤT CẢ các checkpoints trong outputs/papers/siglip2-confusion-seed0_v3/checkpoints/
#    (Gồm: best/, best_loss/, best_macro_f1/)
# 2. Với MỖI checkpoint:
#    - Forward GPU trích xuất Logits ảnh gốc + ảnh lật trên VALIDATION.
#    - Sweep TTA (w_orig, w_flip) trên VALIDATION để tìm trọng số tối ưu.
#    - Áp dụng bộ trọng số tối ưu từ Val để đánh giá trên TEST (Zero Leakage).
# 3. In bảng tổng sắp so sánh toàn bộ các checkpoint (Accuracy, F1, TTA gain).
# 4. Tìm kiếm tổ hợp Ensemble tối ưu nhất (Combinatorial Ensemble).
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

# Thiết lập CUDA / cuDNN libraries cho V100 GPU
SITE_PACKAGES=$("$FER_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
NVIDIA_LIB="$SITE_PACKAGES/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo "=========================================================================="
echo " VALIDATION-TUNED TTA SWEEP & ENSEMBLE TRÊN GPU V100 (FER2013 SEED 0 V3)"
echo " Config         : $CONFIG"
echo " Exp Dir        : $EXP_DIR"
echo " Job ID         : ${SLURM_JOB_ID:-standalone}"
echo " Start Time     : $(date)"
echo "=========================================================================="

nvidia-smi

# Chạy TTA Sweep cho từng checkpoint + Ensemble trên GPU V100
"$FER_PY" -u scripts/sweep_tta_and_ensemble_all_checkpoints.py \
    --config "$CONFIG" \
    --exp-dir "$EXP_DIR" \
    --step 0.05 \
    --include-best-loss \
    --include-best-macro-f1 \
    --num-comb-samples 50000 \
    --save-individual \
    --output "$EXP_DIR/val_tuned_tta_ensemble_report.json"

echo ""
echo "=========================================================================="
echo " [FINISHED] TTA Sweep & Ensemble trên GPU hoàn thành lúc $(date)"
echo " Báo cáo tổng hợp đã lưu tại: $EXP_DIR/val_tuned_tta_ensemble_report.json"
echo "=========================================================================="
