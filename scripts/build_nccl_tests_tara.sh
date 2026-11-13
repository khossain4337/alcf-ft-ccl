#!/bin/bash
# Build nccl-tests (MPI) against our NCCL on Tara-North. Run on a compute node.
# Usage: bash build_nccl_tests_tara.sh [2.21.1] [2.32.6-1]
set -euo pipefail
TVER=${1:-2.21.1}
NVER=${2:-2.32.6-1}
SOFT=/vast/draco/tara/projects/Tara_Deployment/software
SRC=$SOFT/repositories/nccl-tests_v$TVER/nccl-tests
OUT=$SRC/../build_nccl${NVER}_cu13.0
LOG=$SOFT/testing/nccl_tests_build_logs/v$TVER/build_nccl${NVER}_$(date +%Y%m%d-%H%M%S).log
CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/13.0
NCCL_HOME=$SOFT/repositories/nccl_$NVER/build_cu13.0
MPI_HOME=/opt/cray/pe/mpich/9.1.0/ofi/nvidia/23.3
CXX=/opt/cray/pe/gcc-native/14/bin/g++

mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1
cd "$SRC"
make -j 32 MPI=1 MPI_HOME=$MPI_HOME CUDA_HOME=$CUDA_HOME NCCL_HOME=$NCCL_HOME \
  CXX=$CXX BUILDDIR="$OUT" NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
ls "$OUT"
