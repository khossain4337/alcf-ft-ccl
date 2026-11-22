#!/bin/bash
# Build + run cooperative replace on all allocated nodes (2+). Victim = position (default 1).
# Usage: bash run_replace_cooperative_tara.sh [victim] [socket] [ofi]   (default nets: both)
S=$(cd "$(dirname "$0")" && pwd)
SOFT=/vast/draco/tara/projects/Tara_Deployment/software
BIN=$S/../build/replace_cooperative
V=1; [[ $1 =~ ^[0-9]+$ ]] && { V=$1; shift; }
bash $S/build_replace_cooperative_tara.sh || exit 1
NN=$(sort -u $PBS_NODEFILE | wc -l)
for NET in ${@:-socket ofi}; do
  OUT=$SOFT/testing/replace_cooperative/${NET}_${NN}n_v${V}_$(date +%Y%m%d-%H%M%S); mkdir -p $OUT
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT NP=$((4*NN)) NET=$NET \
    bash $S/functionality/env_tara.sh $BIN $OUT $V > $OUT/run.log 2>&1
  echo "== $NET victim $V rc=$? $OUT"
  { grep -m1 'NCCL version' $OUT/run.log
    grep -m1 'Using network' $OUT/run.log
    grep -E '^(pos|spare) ' $OUT/run.log | sort -n -k2
    cat $OUT/map; } | tee $OUT/summary.txt
done
