#!/bin/bash
#SBATCH --job-name=code-prep
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/code_prep_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/code_prep_%j.err
#
# Extract code vectors and assemble all four experiment datasets.
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
D=$PROJ/data/coderec

# ---- exp1: reconstruction from CSN (diverse) -------------------------------
python scripts/extract_code_vectors.py --input $D/csn_python_short.parquet \
       --output $D/vec_csn_code.parquet --mode code --batch-size 24
python scripts/build_code_sft.py --source $D/csn_python_short.parquet \
       --vectors $D/vec_csn_code.parquet --exp exp1 --out-prefix $D/exp1

# ---- exp1 functional slice: MBPP correct programs (has tests) ---------------
python - <<'PY'
import pandas as pd
m = pd.read_parquet('data/repair/mbpp_mutants.parquet')
u = m.drop_duplicates(subset=['task_id'])[['task_id','spec','correct_code','tests','test_setup']]
u = u.rename(columns={'correct_code':'code'})
u.to_parquet('data/coderec/mbpp_programs.parquet', index=False)
print('mbpp unique programs:', len(u))
PY
python scripts/extract_code_vectors.py --input $D/mbpp_programs.parquet \
       --output $D/vec_mbpp_code.parquet --mode code --batch-size 24
python scripts/build_code_sft.py --source $D/mbpp_programs.parquet \
       --vectors $D/vec_mbpp_code.parquet --exp exp1 --out-prefix $D/exp1_mbpp --eval-frac 0.5

# ---- exp2: repair -----------------------------------------------------------
python - <<'PY'
import pandas as pd
m = pd.read_parquet('data/repair/mbpp_mutants.parquet')
m = m.rename(columns={'buggy_code':'code'})
m[['task_id','spec','code','correct_code','tests','test_setup','bug_type']].to_parquet(
    'data/coderec/repair_pairs.parquet', index=False)
print('repair pairs:', len(m), 'programs:', m.task_id.nunique())
PY
python scripts/extract_code_vectors.py --input $D/repair_pairs.parquet \
       --output $D/vec_buggy.parquet --mode code --batch-size 24
python scripts/extract_code_vectors.py --input $D/repair_pairs.parquet \
       --output $D/vec_spec_buggy.parquet --mode spec_code --batch-size 24

python scripts/build_code_sft.py --source $D/repair_pairs.parquet \
       --vectors $D/vec_buggy.parquet     --exp exp2a --out-prefix $D/exp2a
python scripts/build_code_sft.py --source $D/repair_pairs.parquet \
       --vectors $D/vec_buggy.parquet     --exp exp2b --out-prefix $D/exp2b
python scripts/build_code_sft.py --source $D/repair_pairs.parquet \
       --vectors $D/vec_spec_buggy.parquet --exp exp2d --out-prefix $D/exp2d

echo "PREP DONE"; ls -la $D/*.parquet | awk '{print $5, $9}'
