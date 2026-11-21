#!/bin/bash
# Usage: [NP=n VAR=..] bash env_tara.sh <nccl-tests binary> [args]; env stays in this process.
SOFT=/vast/draco/tara/projects/Tara_Deployment/software
CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/13.0
export LD_LIBRARY_PATH=$SOFT/repositories/nccl_2.32.6-1/build_cu13.0/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
B=$SOFT/repositories/nccl-tests_v2.21.1/build_nccl2.32.6-1_cu13.0
mpiexec -n ${NP:-1} --ppn 4 --envall $B/"$@"
