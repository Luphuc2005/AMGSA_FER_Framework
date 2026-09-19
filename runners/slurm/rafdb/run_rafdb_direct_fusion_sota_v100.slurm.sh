#!/bin/bash
#SBATCH --job-name=RAFDB_DIRECT_SOTA
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_DIRECT_SOTA_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_DIRECT_SOTA_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/rafdb_direct_fusion_sota_acc

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_rafdb_direct_fusion_sota_acc.yaml"

echo "============================================================"
echo " RAF-DB Direct Global + Regional Fusion SOTA Accuracy"
echo " Config: $CONFIG"
echo " Output: outputs/rafdb_direct_fusion_sota_acc"
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

# 1. Verify contract before starting training
echo "============================================================"
echo " Step 1: Running Contract & Sanity Verification..."
echo "============================================================"
"$FER_PY" -u scripts/check_rafdb_direct_fusion_contract.py --config "$CONFIG"

# 2. Train Model
echo "============================================================"
echo " Step 2: Training RAF-DB Direct Fusion SOTA Model..."
echo "============================================================"
"$FER_PY" -u train.py --config "$CONFIG" --no-auto-increment

# 3. Comprehensive 15-Checkpoint TTA Sweep & Combinatorial Ensemble
echo "============================================================"
echo " Step 3: Running TTA Sweep & Ensemble on Checkpoints..."
echo "============================================================"
"$FER_PY" -u scripts/sweep_tta_and_ensemble_all_checkpoints.py --config "$CONFIG" || true

echo "============================================================"
echo " Completed Training & Evaluation Pipeline"
echo " End: $(date)"
echo "============================================================"
