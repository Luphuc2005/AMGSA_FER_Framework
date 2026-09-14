#!/bin/bash
#SBATCH --job-name=RAFDB_CLEAN_FUSION_91
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_CLEAN_FUSION_91_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_CLEAN_FUSION_91_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/rafdb_direct_fusion_clean_91

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
BASELINE="$ROOT/config_rafdb_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"
CONFIG="$ROOT/config_rafdb_direct_fusion_clean_91.yaml"

echo "============================================================"
echo " RAF-DB Clean Baseline 91% + Direct Global-Regional Fusion"
echo " Config:   $CONFIG"
echo " Baseline: $BASELINE"
echo " Output:   outputs/rafdb_direct_fusion_clean_91"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "============================================================"

nvidia-smi

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

# 1. Print Diff & Verify Contract with 91% Baseline
echo "============================================================"
echo " Step 1: Inspecting Config Diff with 91% Baseline..."
echo "============================================================"
"$FER_PY" -u scripts/verify_and_diff_rafdb_clean_candidate.py --baseline "$BASELINE" --candidate "$CONFIG"

# 2. Train Model
echo "============================================================"
echo " Step 2: Training Clean Direct Fusion Model..."
echo "============================================================"
"$FER_PY" -u train.py --config "$CONFIG" --no-auto-increment

# 3. Comprehensive 15-Checkpoint TTA Sweep & Single Checkpoint Evaluation
echo "============================================================"
echo " Step 3: Running Checkpoint Discovery & Evaluation..."
echo "============================================================"
"$FER_PY" -u scripts/sweep_tta_and_ensemble_all_checkpoints.py --config "$CONFIG" || true

echo "============================================================"
echo " Completed Training & Evaluation Pipeline"
echo " End: $(date)"
echo "============================================================"
