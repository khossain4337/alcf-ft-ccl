#!/bin/bash
# Build + run TP+DP crash replace on all allocated nodes (2+).
# Usage: bash run_replace_tpdp_tara.sh [victim] [revoke|abort] [deadline] [mainthread|fullqueue] [socket] [ofi]
#   (defaults: 1 revoke 10 mainthread, both nets)
S=$(cd "$(dirname "$0")" && pwd)
SOFT=/vast/draco/tara/projects/Tara_Deployment/software
BIN=$S/../build/replace_tpdp
V=1; [[ $1 =~ ^[0-9]+$ ]] && { V=$1; shift; }
C=revoke; [[ $1 = revoke || $1 = abort ]] && { C=$1; shift; }
DL=10; [[ $1 =~ ^[0-9]+$ ]] && { DL=$1; shift; }
K=mainthread; [[ $1 = mainthread || $1 = fullqueue ]] && { K=$1; shift; }
bash $S/build_replace_tpdp_tara.sh || exit 1
NN=$(sort -u $PBS_NODEFILE | wc -l)
for NET in ${@:-socket ofi}; do
  OUT=$SOFT/testing/replace_tpdp/${NET}_${NN}n_v${V}_${K}_${C}_$(date +%Y%m%d-%H%M%S); mkdir -p $OUT
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV NCCL_DEBUG_FILE=$OUT/nccl.%h.%p NCCL_CUMEM_ENABLE=1 \
    MPIEXEC_ARGS=--no-abort-on-failure NP=$((4*NN)) NET=$NET \
    timeout 300 bash $S/functionality/env_tara.sh $BIN $OUT $V $C $DL $K > $OUT/run.log 2>&1
  echo "== $NET victim $V $C deadline $DL $K rc=$? (nonzero expected: a rank died) $OUT"
  { awk '/NCCL version/{print; exit}' $OUT/nccl.*  # awk, not cat|grep -m1: SIGPIPE is ignored in jobs
    awk '/Using network/{print; exit}' $OUT/nccl.*
    grep -E '^(pos|spare) |\.(cpp|h):[0-9]+ ' $OUT/run.log | sort -n -k2
    grep -E '_REORDER ' $OUT/run.log | sort
    echo "exclude: $(cat $OUT/exclude)"
    for f in $OUT/*.fail $OUT/*.built.*; do [ -e $f ] && echo "${f##*/}: $(cat $f)"; done; } | tee $OUT/summary.txt
done
