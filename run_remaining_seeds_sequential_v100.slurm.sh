#!/bin/bash
#SBATCH --job-name=FER_SIGLIP2_REMAINING_SEEDS
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_SIGLIP2_REMAINING_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_SIGLIP2_REMAINING_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=1
export TF_CUDNN_USE_AUTOTUNE=1
export TF_ENABLE_CUBLAS_TENSOR_OP_MATH=1
export TF_ENABLE_CUDNN_TENSOR_OP_MATH=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

echo "============================================================"
echo " FER2013 ConvNeXt-Base MS1M SigLIP 2 Confusion"
echo " Sequential Runner for Remaining Seeds (GPU V100)"
echo " Seeds 0, 1, 42 already completed."
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "============================================================"

nvidia-smi

# Danh sách các seed còn lại: 43, 123, 3047, 777, 2024, 3407
# (Nếu chỉ muốn chạy thêm 2 seed để hoàn thành bộ 5-seed: chỉnh SEEDS=(43 123))
SEEDS=(43 123 3047 777 2024 3407)

for SEED in "${SEEDS[@]}"; do
    CONFIG="$ROOT/config_convnext_base_ms1m_adaptive_siglip2_confusion_seed${SEED}.yaml"
    OUTPUT_DIR="outputs/papers/siglip2-confusion-seed${SEED}"
    
    echo "============================================================"
    echo " >>> STARTING PIPELINE FOR SEED: $SEED <<<"
    echo " Config: $CONFIG"
    echo " Output: $OUTPUT_DIR"
    echo " Time:   $(date)"
    echo "============================================================"

    mkdir -p "$OUTPUT_DIR"

    # 1. Train Model
    "$FER_PY" -u train.py --config "$CONFIG"

    # 2. Automated TTA Sweep on Best Accuracy Checkpoint
    echo "============================================================"
    echo " Running Automated TTA Sweep on Best Accuracy Checkpoint (Seed $SEED)..."
    echo "============================================================"
    "$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --step 0.05 || true

    # 3. Automated TTA Sweep on Best Loss Checkpoint
    echo "============================================================"
    echo " Running Automated TTA Sweep on Best Loss Checkpoint (Seed $SEED)..."
    echo "============================================================"
    BEST_LOSS_CKPT=$(ls -t "$ROOT"/outputs/papers/siglip2-confusion-seed${SEED}*/checkpoints/best_loss/ckpt-*.index 2>/dev/null | head -n 1 | sed 's/\.index$//' || true)
    if [ -n "$BEST_LOSS_CKPT" ]; then
        "$FER_PY" -u sweep_tta_weights.py --config "$CONFIG" --checkpoint "$BEST_LOSS_CKPT" --step 0.05 || true
    fi

    # 4. Automated Top-5 Checkpoint Softmax Ensemble + TTA Evaluation
    echo "============================================================"
    echo " Running Automated Top-5 Ensemble + TTA (Seed $SEED)..."
    echo "============================================================"
    if [ -f "scripts/evaluate_top5_ensemble_siglip2.py" ]; then
        "$FER_PY" -u scripts/evaluate_top5_ensemble_siglip2.py --config "$CONFIG" || true
    fi

    # 5. Comprehensive Checkpoint TTA Sweep & Combinatorial Ensemble
    echo "============================================================"
    echo " Running Full Checkpoint TTA Sweep & Ensemble (Seed $SEED)..."
    echo "============================================================"
    "$FER_PY" -u scripts/sweep_tta_and_ensemble_all_checkpoints.py --config "$CONFIG" || true

    echo ">>> Finished SEED $SEED at $(date) <<<"
done

echo "============================================================"
echo " All Remaining Seeds Finished Successfully!"
echo " End: $(date)"
echo "============================================================"
