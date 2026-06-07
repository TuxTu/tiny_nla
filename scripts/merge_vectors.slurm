#!/bin/bash
#SBATCH --job-name=nla-merge
#SBATCH --partition=berzelius-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/merge_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/merge_%j.err

set -euo pipefail

CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
DATA=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/data

echo "=== Merge $PREFIX shards → $OUTPUT ==="

python -c "
import pyarrow.parquet as pq, pyarrow as pa
from pathlib import Path
tables = []
for i in range($NUM_SHARDS):
    shard = f'$DATA/${OUTPUT}.shard_{i:03d}'
    t = pq.read_table(shard)
    tables.append(t)
    print(f'  read {shard}: {t.num_rows} rows')
    Path(shard).unlink()
merged = pa.concat_tables(tables)
pq.write_table(merged, '$DATA/$OUTPUT')
print(f'merged {merged.num_rows} rows → $DATA/$OUTPUT')
"

echo "=== Merge done ==="
ls -lh "$DATA/$OUTPUT"
