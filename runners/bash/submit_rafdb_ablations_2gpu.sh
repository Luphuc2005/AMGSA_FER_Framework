#!/bin/bash
# Submit RAF-DB Ablations across 2 GPUs in parallel

set -euo pipefail

echo "============================================================"
echo " Submitting RAF-DB Ablation Study across 2 GPUs..."
echo " GPU 1 (Worker 1): Stages 1, 2, 3"
echo " GPU 2 (Worker 2): Stages 4, 5"
echo "============================================================"

JOB1=$(sbatch run_rafdb_ablations_2gpu_part1.slurm.sh | awk '{print $4}')
echo " Worker 1 (Stages 1->2->3) submitted: Job ID $JOB1"

JOB2=$(sbatch run_rafdb_ablations_2gpu_part2.slurm.sh | awk '{print $4}')
echo " Worker 2 (Stages 4->5)    submitted: Job ID $JOB2"

echo "============================================================"
echo " Both workers are now running concurrently on 2 GPUs!"
echo " Monitor queue:  squeue -u $USER"
echo " Worker 1 logs:  tail -f logs/RAFDB_2G_P1_$JOB1.out"
echo " Worker 2 logs:  tail -f logs/RAFDB_2G_P2_$JOB2.out"
echo "============================================================"