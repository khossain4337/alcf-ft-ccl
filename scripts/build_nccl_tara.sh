#!/bin/bash
# Build NCCL from source on Tara-North (GH200). Run on a compute node.
# Usage: bash build_nccl_tara.sh [2.32.6-1]
set -euo pipefail
VER=${1:-2.32.6-1}
SOFT=/vast/draco/tara/projects/Tara_Deployment/software
SRC=$SOFT/repositories/nccl_$VER/nccl
OUT=$SRC/../build_cu13.0
LOG=$SOFT/testing/nccl_build_logs/v$VER/build_cu13.0_$(date +%Y%m%d-%H%M%S).log
CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/13.0
CXX=/opt/cray/pe/gcc-native/14/bin/g++

mkdir -p "$(dirname "$LOG")"
cd "$SRC"
make -j 32 src.build CUDA_HOME=$CUDA_HOME CXX=$CXX BUILDDIR="$OUT" \
  NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90" 2>&1 | tee "$LOG" | tail -20
ls "$OUT/lib"
grep -c ncclCommGrow "$OUT/include/nccl.h"
