#!/bin/bash
# nccl-tests on our NCCL 2.32.6. Run inside an allocation; node count from $PBS_NODEFILE.
# Usage: bash run_nccl_tests_tara.sh socket|ofi [tests...]   (default: all_reduce)
NET=$1; shift; TESTS=${*:-all_reduce}
SOFT=/vast/draco/tara/projects/Tara_Deployment/software
B=$SOFT/repositories/nccl-tests_v2.21.1/build_nccl2.32.6-1_cu13.0
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
OUT=$SOFT/testing/nccl_tests_runs/${NET}_${NN}n_$(date +%Y%m%d-%H%M%S); mkdir -p $OUT; echo $OUT
{ ldd $B/all_reduce_perf | grep -E 'nccl|mpi|not found'
  [ -n "$NCCL_NET_PLUGIN" ] && ldd $NCCL_NET_PLUGIN | grep -E 'libfabric|not found'; } | tee $OUT/ldd.txt
for t in $TESTS; do
  mpiexec -n $((4*NN)) --ppn 4 --envall $VNI --cpu-bind list:0-71:72-143:144-215:216-287 \
    $B/${t}_perf -b 8 -e 4G -f 2 -g 1 -n 50 -w 10 2>&1 | tee $OUT/$t.log
done
for f in $OUT/*.log; do echo "== $(basename $f .log)"; grep -m1 'NCCL version' $f
  grep -m2 -E 'Loaded net plugin|Using network' $f; grep -E '^ +4294967296 |Out of bounds|Avg bus' $f
done | tee $OUT/summary.txt
