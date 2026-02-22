#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Updating apt indexes..."
sudo apt-get update

echo "[2/6] Installing core build dependencies..."
sudo apt-get install -y \
  build-essential \
  cmake \
  ninja-build \
  pkg-config \
  python3-dev \
  python3-venv \
  python3-pip \
  wget \
  ca-certificates \
  gnupg

echo "[3/6] Configuring NVIDIA CUDA apt repo for Ubuntu 24.04 (if not present)..."
if ! ls /etc/apt/sources.list.d/*cuda* >/dev/null 2>&1; then
  tmp_deb="/tmp/cuda-keyring_1.1-1_all.deb"
  wget -O "${tmp_deb}" "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
  sudo dpkg -i "${tmp_deb}"
  rm -f "${tmp_deb}"
fi

echo "[4/6] Installing CUDA toolkit..."
sudo apt-get update
sudo apt-get install -y cuda-toolkit

echo "[5/6] Ensuring PATH for nvcc in current shell..."
if ! command -v nvcc >/dev/null 2>&1; then
  for candidate in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
    if [ -d "${candidate}" ]; then
      export PATH="${candidate}:${PATH}"
      break
    fi
  done
fi

echo "[6/6] Verifying CUDA toolchain..."
command -v nvcc
nvcc --version
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv,noheader

echo "Setup complete."
