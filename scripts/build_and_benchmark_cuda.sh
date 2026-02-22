#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

. .venv/bin/activate

echo "[1/5] Installing Python build deps..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade numpy pytest scikit-build-core pybind11

echo "[1.5/5] Resolving nvcc path..."
if command -v nvcc >/dev/null 2>&1; then
  NVCC_PATH="$(command -v nvcc)"
elif [ -x "/usr/local/cuda/bin/nvcc" ]; then
  NVCC_PATH="/usr/local/cuda/bin/nvcc"
  export PATH="/usr/local/cuda/bin:${PATH}"
elif [ -x "/usr/local/cuda-13.1/bin/nvcc" ]; then
  NVCC_PATH="/usr/local/cuda-13.1/bin/nvcc"
  export PATH="/usr/local/cuda-13.1/bin:${PATH}"
else
  echo "nvcc not found. Run scripts/wsl2_cuda_setup.sh first."
  exit 1
fi
export CUDACXX="${NVCC_PATH}"
echo "Using nvcc: ${NVCC_PATH}"
nvcc --version
echo "CUDA include dir check:"
ls -la /usr/local/cuda/include/cuda_runtime.h || true
echo "CUDA lib dir check:"
ls -la /usr/local/cuda/lib64/libcudart.so* || true

echo "[2/5] Detecting compute capability..."
CC_RAW="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n 1 | tr -d '\r')"
if [ -z "${CC_RAW}" ]; then
  echo "Could not read compute capability from nvidia-smi."
  exit 1
fi
CUDA_ARCH="$(echo "${CC_RAW}" | tr -d '.')"
echo "Detected compute capability: ${CC_RAW} (arch token ${CUDA_ARCH})"
echo "Using CMake CUDA architectures: native"

echo "[3/5] Building editable package + CUDA extension..."
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CUDA_COMPILER=${NVCC_PATH}" \
  python -m pip install -e .

echo "[4/5] Running tests..."
pytest -q

echo "[5/5] Running backend benchmark..."
python scripts/benchmark_backends.py --n-sims 200000 --num-points 21 --chunk-size 50000 --sample-size 50000
