#!/bin/bash
# Build src/nompi_probe.cpp (no MPI) against our NCCL. Usage: bash build_nompi_tara.sh
set -euo pipefail
SOFT=/vast/draco/tara/projects/Tara_Deployment/software
CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/13.0
NCCL_HOME=$SOFT/repositories/nccl_2.32.6-1/build_cu13.0
D=$(cd "$(dirname "$0")/.." && pwd); mkdir -p $D/build
/opt/cray/pe/gcc-native/14/bin/g++ -std=c++17 -O2 -Wall $D/src/nompi_probe.cpp -o $D/build/nompi_probe \
  -I$CUDA_HOME/include -I$NCCL_HOME/include -L$CUDA_HOME/lib64 -L$NCCL_HOME/lib -lnccl -lcudart
ls -l $D/build/nompi_probe
