#!/bin/bash
#SBATCH --job-name=FER_DUAL_SMIRK_ATT
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --nodelist=gpunode1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_DUAL_SMIRK_ATT_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/FER_DUAL_SMIRK_ATT_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/dual_convnext_smirk_guided_attention/logs outputs/dual_convnext_smirk_guided_attention/checkpoints

export PYTHONUNBUFFERED=1

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"
CONFIG="$ROOT/config_dual_convnext_smirk_guided_attention.yaml"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "============================================================"
echo " FROZEN RGB CKPT-43 + SMALL SMIRK GEOMETRY ATTENTION (V100)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-standalone}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Start: $(date)"
echo
echo "ROOT=$ROOT"
echo "FER_PY=$FER_PY"
echo "CONFIG=$CONFIG"
echo "Stage 1 contract: RGB frozen, SAM disabled, mixed precision enabled, grad clip=1.0"
echo "============================================================"

nvidia-smi

[ -x "$FER_PY" ] || { echo "[ERROR] FER TensorFlow python not found: $FER_PY"; exit 1; }
[ -f "$CONFIG" ] || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }

# Configure NVIDIA CUDA/CUDNN libraries for TensorFlow GPU
export NVIDIA_LIB=/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/lib/python3.9/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:${LD_LIBRARY_PATH:-}"

echo
echo "============================================================"
echo " STARTING STAGE 1 SMIRK GEOMETRY-GUIDED ATTENTION TRAINING"
echo "============================================================"

"$FER_PY" -u scripts/train_dual_convnext_smirk_guided_attention.py \
    --config "$CONFIG"

echo
echo "============================================================"
echo " STAGE 1 DUAL SMIRK ATTENTION TRAINING COMPLETED"
echo " End: $(date)"
echo "============================================================"
