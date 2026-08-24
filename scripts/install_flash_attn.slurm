#!/bin/bash
#SBATCH --job-name=install-fa
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-40GB:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/install_fa_%j.out
#SBATCH --error=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/logs/install_fa_%j.err
set -euo pipefail

# Load build environment (gcc, g++, make) and CUDA toolkit
source /etc/profile.d/modules.sh 2>/dev/null || true
module load buildenv-gcc/2018a-eb 2>/dev/null || true
module load CUDA/12.2.0 2>/dev/null || true
module load cuda/12.2 2>/dev/null || true
module load cuda 2>/dev/null || true

# Find CUDA
for c in /usr/local/cuda-12* /usr/local/cuda /software/cuda/* /proj/*/cuda*; do
    if [ -f "$c/bin/nvcc" ]; then
        export CUDA_HOME="$c"
        export PATH="$CUDA_HOME/bin:$PATH"
        break
    fi
done

# If still not found, search
if [ -z "${CUDA_HOME:-}" ]; then
    nvcc_path=$(find /usr/local /software -name nvcc -type f 2>/dev/null | head -1)
    if [ -n "$nvcc_path" ]; then
        export CUDA_HOME=$(dirname $(dirname "$nvcc_path"))
        export PATH="$CUDA_HOME/bin:$PATH"
    fi
fi

echo "CUDA_HOME: ${CUDA_HOME:-NOT FOUND}"
echo "nvcc: $(which nvcc 2>/dev/null || echo 'NOT FOUND')"
echo "g++: $(which g++ 2>/dev/null || echo 'NOT FOUND')"

CONDA_ENV=/home/x_tuhan/.conda/envs/tiny_nla
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export PIP_CACHE_DIR=/proj/assert-berzelius/users/x_tuhan/garage/tiny_nla/cache/pip
mkdir -p "$PIP_CACHE_DIR"

# Try pre-built wheel first (much faster)
echo "Trying pre-built wheel..."
pip install --cache-dir "$PIP_CACHE_DIR" flash-attn --no-build-isolation 2>&1 || {
    echo "Pre-built failed, trying specific version..."
    pip install --cache-dir "$PIP_CACHE_DIR" flash-attn==2.7.4.post1 --no-build-isolation 2>&1
}

echo "=== Done ==="
python -c "import flash_attn; print('flash_attn OK:', flash_attn.__version__)" 2>&1 || echo "WARNING: flash_attn import failed"
