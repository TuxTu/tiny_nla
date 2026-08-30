#!/bin/bash
#SBATCH --job-name=code-exp
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:4
#SBATCH --ntasks=1 --cpus-per-task=32 --mem=400G
#SBATCH --time=01:45:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/code_%x_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/code_%x_%j.err
#
# Train + evaluate one code experiment. EXP is set by the driver.
#   exp1  E(code)         -> code      reconstruction; gates everything else
#   exp2a E(buggy)        -> correct   repair, no spec
#   exp2b E(buggy)+spec   -> correct   spec as prompt text; PREDICTED TO COLLAPSE
#   exp2d E(spec++buggy)  -> correct   joint encoding
set -euo pipefail
CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
PROJ=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla
export HF_HOME=$PROJ/cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=$PROJ/cache/torchinductor
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJ"

EXP=${EXP:?set EXP}
D=$PROJ/data/coderec
OUT=$PROJ/checkpoints/code_${EXP}
SUM=$PROJ/logs/code_experiments_summary.md

echo "=== $EXP train ==="
torchrun --nproc_per_node=4 --master_port=$((20000 + RANDOM % 10000)) \
  -m nla.training.train_actor_sft \
    --data "$D/${EXP}_train.parquet" --eval-data "$D/${EXP}_eval.parquet" \
    --model-name Qwen/Qwen3-8B --output-dir "$OUT" \
    --injection-scale 300 \
    --global-batch-size 256 --micro-batch-size 4 \
    --num-steps 250 --lr 2e-5 \
    --eval-every 50 --eval-samples 512 --eval-batch-size 8 --eval-gap \
    --save-every 250 --fsdp

echo "=== $EXP generation eval ==="
python scripts/eval_code_recon.py --actor-ckpt "$OUT" \
    --data "$D/${EXP}_eval.parquet" --n-samples 300 \
    --out "$PROJ/logs/code_${EXP}_gen.jsonl" || echo "(gen eval failed)"
python scripts/eval_code_recon.py --actor-ckpt "$OUT" \
    --data "$D/${EXP}_eval.parquet" --n-samples 300 --shuffle-control \
    --out "$PROJ/logs/code_${EXP}_gen_shuf.jsonl" || echo "(shuf eval failed)"

{
  echo ""
  echo "## $EXP  (job $SLURM_JOB_ID, $(date '+%a %H:%M'))"
  echo '```'
  grep "\[eval\]" "$PROJ/logs/code_${EXP}_${SLURM_JOB_ID}.out" 2>/dev/null | tail -6 || \
    grep -h "\[eval\]" $PROJ/logs/code_*${SLURM_JOB_ID}.out 2>/dev/null | tail -6
  echo "--- generation (correct vectors) ---"
  python -c "import json,sys; d=json.loads(open('$PROJ/logs/code_${EXP}_gen.jsonl').readline()); print('  extracted %d/%d  exact %.1f%%  ast-equal %.1f%%  tok-sim %.3f'%(d['extracted'],d['n'],100*d['exact']/max(d['extracted'],1),100*d['ast_equal']/max(d['extracted'],1),d['tok_sim_mean'] or 0))" 2>/dev/null || echo "  (unavailable)"
  echo "--- generation (SHUFFLED control) ---"
  python -c "import json,sys; d=json.loads(open('$PROJ/logs/code_${EXP}_gen_shuf.jsonl').readline()); print('  extracted %d/%d  exact %.1f%%  ast-equal %.1f%%  tok-sim %.3f'%(d['extracted'],d['n'],100*d['exact']/max(d['extracted'],1),100*d['ast_equal']/max(d['extracted'],1),d['tok_sim_mean'] or 0))" 2>/dev/null || echo "  (unavailable)"
  echo '```'
} >> "$SUM"
echo "$EXP DONE -> $SUM"
