#!/bin/bash
# Submit the whole code-experiment chain with Slurm dependencies, so the
# CLUSTER runs the sequence autonomously -- it does not depend on any
# interactive session staying alive.
set -euo pipefail
cd /proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
S=$(sbatch --parsable scripts/code_exp_prep.slurm)
echo "prep      $S"
PREV=$S
for E in exp1 exp2a exp2b exp2d; do
  J=$(sbatch --parsable --dependency=afterok:$PREV --job-name="$E" \
        --export=ALL,EXP=$E scripts/code_exp_train.slurm)
  echo "$E       $J  (after $PREV)"
  PREV=$J
done
echo
echo "results accumulate in logs/code_experiments_summary.md"
