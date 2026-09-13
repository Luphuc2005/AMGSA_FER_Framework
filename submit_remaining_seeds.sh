#!/bin/bash
# Helper script to submit remaining seed jobs to SLURM cluster
# Seeds 0, 1, 42 are already completed.

set -euo pipefail

echo "============================================================"
echo " Submitting Remaining Seeds to SLURM (GPU V100)"
echo " Seeds 0, 1, 42 are already completed."
echo "============================================================"

# Mode selection:
# Run: ./submit_remaining_seeds.sh [5seeds | all]
# Default is 'all' remaining seeds (43, 123, 3047, 777, 2024, 3407)
# Use '5seeds' if you only want seeds 43 and 123 (to complete 5 seeds: 0, 1, 42, 43, 123)

MODE="${1:-all}"

if [ "$MODE" = "5seeds" ]; then
    echo "Mode: Completing 5-seed benchmark (seeds 43, 123)..."
    SEEDS=(43 123)
else
    echo "Mode: Running all remaining seeds (43, 123, 3047, 777, 2024, 3407)..."
    SEEDS=(43 123 3047 777 2024 3407)
fi

for SEED in "${SEEDS[@]}"; do
    SCRIPT="run_siglip2_confusion_seed${SEED}_v100.slurm.sh"
    if [ -f "$SCRIPT" ]; then
        JOB_ID=$(sbatch "$SCRIPT" | awk '{print $4}')
        echo " [+] Seed $SEED submitted: Job ID $JOB_ID ($SCRIPT)"
    else
        echo " [!] Script $SCRIPT not found, skipping."
    fi
done

echo "============================================================"
echo " Submission complete!"
echo " Check running jobs with: squeue -u \$USER"
echo "============================================================"
