#!/bin/bash
# Build src/allreduce.cpp with g++ 14 against our NCCL. Usage: bash build_allreduce_tara.sh
set -euo pipefail
SOFT=/vast/draco/tara/projects/Tara_Deployment/software
CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/13.0
NCCL_HOME=$SOFT/repositories/nccl_2.32.6-1/build_cu13.0
MPI_HOME=/opt/cray/pe/mpich/9.1.0/ofi/nvidia/23.3
D=$(cd "$(dirname "$0")/.." && pwd); mkdir -p $D/build
/opt/cray/pe/gcc-native/14/bin/g++ -std=c++17 -O2 -Wall $D/src/allreduce.cpp -o $D/build/allreduce \
  -I$CUDA_HOME/include -I$NCCL_HOME/include -I$MPI_HOME/include \
  -L$CUDA_HOME/lib64 -L$NCCL_HOME/lib -L$MPI_HOME/lib -lnccl -lcudart -lmpi
ls -l $D/build/allreduce
