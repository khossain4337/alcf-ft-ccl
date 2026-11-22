#!/bin/bash
# Build + run the no-MPI probe on all allocated nodes (2+; no --single-node-vni).
# Usage: bash run_nompi_tara.sh [socket] [ofi]   (default: both)
S=$(cd "$(dirname "$0")" && pwd)
SOFT=/vast/draco/tara/projects/Tara_Deployment/software
BIN=$S/../build/nompi_probe
bash $S/build_nompi_tara.sh || exit 1
ldd $BIN | grep -i mpi && { echo "FAIL: binary links MPI"; exit 1; }
NN=$(sort -u $PBS_NODEFILE | wc -l)
for NET in ${@:-socket ofi}; do
  OUT=$SOFT/testing/nompi/${NET}_${NN}n_$(date +%Y%m%d-%H%M%S); mkdir -p $OUT
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NP=$((4*NN)) NET=$NET \
    bash $S/functionality/env_tara.sh $BIN $OUT > $OUT/run.log 2>&1
  echo "== $NET rc=$? $OUT"
  { grep -m1 'NCCL version' $OUT/run.log
    grep -m2 -E 'Loaded net plugin|Using network' $OUT/run.log
    grep '^rank ' $OUT/run.log | sort -n -k2; } | tee $OUT/summary.txt
done
