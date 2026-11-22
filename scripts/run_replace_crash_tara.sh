#!/bin/bash
# Build + run crash replace on all allocated nodes (2+). Victim = position (default 1), mode kill|hang, deadline s.
# Usage: bash run_replace_crash_tara.sh [victim] [kill|hang] [deadline] [socket] [ofi]   (defaults: 1 kill 10, both nets)
S=$(cd "$(dirname "$0")" && pwd)
SOFT=/vast/draco/tara/projects/Tara_Deployment/software
BIN=$S/../build/replace_crash
V=1; [[ $1 =~ ^[0-9]+$ ]] && { V=$1; shift; }
M=kill; [[ $1 = kill || $1 = hang ]] && { M=$1; shift; }
DL=10; [[ $1 =~ ^[0-9]+$ ]] && { DL=$1; shift; }
bash $S/build_replace_crash_tara.sh || exit 1
NN=$(sort -u $PBS_NODEFILE | wc -l)
for NET in ${@:-socket ofi}; do
  OUT=$SOFT/testing/replace_crash/${NET}_${NN}n_v${V}_${M}_d${DL}_$(date +%Y%m%d-%H%M%S); mkdir -p $OUT
  # NCCL logs per process: PALS interleaves merged stdout mid-line.
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV NCCL_DEBUG_FILE=$OUT/nccl.%h.%p NCCL_CUMEM_ENABLE=1 \
    MPIEXEC_ARGS=--no-abort-on-failure NP=$((4*NN)) NET=$NET \
    timeout 300 bash $S/functionality/env_tara.sh $BIN $OUT $V $M $DL > $OUT/run.log 2>&1
  echo "== $NET victim $V $M deadline $DL rc=$? (nonzero expected: a rank died) $OUT"
  { cat $OUT/nccl.* | grep -m1 'NCCL version'
    cat $OUT/nccl.* | grep -m1 'Using network'
    cat $OUT/nccl.* | grep -m1 'CUMEM'
    grep -E '^(pos|spare) |\.(cpp|h):[0-9]+ ' $OUT/run.log | sort -n -k2
    echo "exclude: $(cat $OUT/exclude)"
    for f in $OUT/shrunk.*; do echo "${f##*/}: $(cat $f)"; done
    cat $OUT/map; } | tee $OUT/summary.txt
done
