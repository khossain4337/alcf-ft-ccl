#!/bin/bash
# Usage: [NP=n] [NET=socket|ofi] bash env_tara.sh <nccl-tests binary | /abs/path> [args]; env stays in this process.
SOFT=/vast/draco/tara/projects/Tara_Deployment/software
CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/13.0
export LD_LIBRARY_PATH=$SOFT/repositories/nccl_2.32.6-1/build_cu13.0/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
B=$SOFT/repositories/nccl-tests_v2.21.1/build_nccl2.32.6-1_cu13.0
if [ "$NET" = ofi ]; then
  export NCCL_NET_PLUGIN=$SOFT/repositories/aws_nccl_ofi_v1.22.0/v1.22.0_lf2.3.1_cu13.0/lib/libnccl-net-ofi.so
  export LD_LIBRARY_PATH=$SOFT/repositories/libfabric_v2.7.0/v2.7.0_cu13.0/lib:$LD_LIBRARY_PATH
  export NCCL_CROSS_NIC=1 NCCL_NET_GDR_LEVEL=PHB
  export FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DISABLE_HOST_REGISTER=1 FI_CXI_DEFAULT_CQ_SIZE=131072 \
    FI_CXI_RDZV_PROTO=alt_read FI_CXI_RDZV_EAGER_SIZE=0 FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_GET_MIN=0 \
    FI_CXI_DEFAULT_TX_SIZE=2048 FI_CXI_ENABLE_WRITEDATA=1
else
  export NCCL_NET=Socket
fi
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
[[ $1 = /* ]] || set -- "$B/$1" "${@:2}"
mpiexec -n ${NP:-1} --ppn 4 --envall "$@"
