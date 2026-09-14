#!/bin/bash
#SBATCH --job-name=RAFDB_2G_P1
#SBATCH --partition=gpu-queue
#SBATCH --account=sokhcn
#SBATCH --qos=gpu-q
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_2G_P1_%j.out
#SBATCH --error=/home/ptbao/projects/FER2013_MGR_CNN/logs/RAFDB_2G_P1_%j.err

set -euo pipefail

ROOT=/home/ptbao/projects/FER2013_MGR_CNN
cd "$ROOT"

mkdir -p logs outputs/ablation/rafdb

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

FER_PY="/home/ptbao/projects/FER2013_MGR_CNN/fer2013_env/bin/python"

echo "============================================================"
echo " RAF-DB 2-GPU WORKER 1: Stages 1 -> 2 -> 3"
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

STAGES=(
  "config_rafdb_ablation_1_baseline.yaml"
  "config_rafdb_ablation_2_siglip2_single_proto.yaml"
  "config_rafdb_ablation_3_siglip2_multigranularity.yaml"
)

for CFG in "${STAGES[@]}"; do
  echo "============================================================"
  echo " [$(date)] STARTING: $CFG"
  echo "============================================================"
  "$FER_PY" -u train.py --config "$ROOT/$CFG"
  echo " [$(date)] Running 15-Checkpoint TTA Sweep & Ensemble..."
  "$FER_PY" -u scripts/sweep_tta_and_ensemble_all_checkpoints.py --config "$ROOT/$CFG" || true
  echo "============================================================"
  echo " [$(date)] FINISHED: $CFG"
  echo "============================================================"
done

echo "============================================================"
echo " WORKER 1 (STAGES 1, 2, 3) FINISHED SUCCESSFULLY!"
echo " End: $(date)"
echo "============================================================"