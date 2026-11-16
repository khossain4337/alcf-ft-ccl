#!/bin/bash
# Own allreduce on our NCCL 2.32.6. Run inside an allocation; node count from $PBS_NODEFILE.
# Usage: bash run_allreduce_tara.sh socket|ofi [count]
NET=$1; COUNT=$2
SOFT=/vast/draco/tara/projects/Tara_Deployment/software
BIN=$(cd "$(dirname "$0")/.." && pwd)/build/allreduce
NN=$(sort -u $PBS_NODEFILE | wc -l)
CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/13.0
export LD_LIBRARY_PATH=$SOFT/repositories/nccl_2.32.6-1/build_cu13.0/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,NET NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
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
VNI=; [ $NN = 1 ] && VNI=--single-node-vni
OUT=$SOFT/testing/allreduce_runs/${NET}_${NN}n_$(date +%Y%m%d-%H%M%S); mkdir -p $OUT; echo $OUT
{ ldd $BIN | grep -E 'nccl|mpi|cudart|not found'
  [ -n "$NCCL_NET_PLUGIN" ] && ldd $NCCL_NET_PLUGIN | grep -E 'libfabric|not found'; } | tee $OUT/ldd.txt
mpiexec -n $((4*NN)) --ppn 4 --envall $VNI --cpu-bind list:0-71:72-143:144-215:216-287 \
  $BIN $COUNT 2>&1 | tee $OUT/allreduce.log
{ grep -m1 'NCCL version' $OUT/allreduce.log
  grep -m2 -E 'Loaded net plugin|Using network' $OUT/allreduce.log
  grep '^allreduce ' $OUT/allreduce.log; } | tee $OUT/summary.txt
