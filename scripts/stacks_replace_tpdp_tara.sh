#!/bin/bash
# Run one TP+DP crash replace and dump every rank's stacks (gdb) once it should be recovering.
# Usage: bash stacks_replace_tpdp_tara.sh [victim] [revoke|abort] [deadline] [socket|ofi] [mainthread|fullqueue] [at s]
#   (defaults: 1 abort 10 socket mainthread; stacks 20 s after exclude, or [at] s after start)
S=$(cd "$(dirname "$0")" && pwd)
SOFT=/vast/draco/tara/projects/Tara_Deployment/software
V=${1:-1} C=${2:-abort} DL=${3:-10} NET=${4:-socket} K=${5:-mainthread} AT=$6
NN=$(sort -u $PBS_NODEFILE | wc -l)
OUT=$SOFT/testing/replace_tpdp/${NET}_${NN}n_v${V}_${K}_${C}_stacks_$(date +%Y%m%d-%H%M%S); mkdir -p $OUT
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,DESTROY NCCL_DEBUG_FILE=$OUT/nccl.%h.%p NCCL_CUMEM_ENABLE=1 \
  MPIEXEC_ARGS=--no-abort-on-failure NP=$((4*NN)) NET=$NET \
  timeout 300 bash $S/functionality/env_tara.sh $S/../build/replace_tpdp $OUT $V $C $DL $K > $OUT/run.log 2>&1 &
JOB=$!
if [ -n "$AT" ]; then sleep $AT; else
  for i in $(seq 120); do [ -e $OUT/exclude ] && break; sleep 1; done
  [ -e $OUT/exclude ] || { echo "no exclude after 120 s; no stacks taken"; wait $JOB; exit 1; }
  sleep 20  # stop + drain take ~1 s on a healthy run; anything still there at 20 s is stuck
fi
for h in $(sort -u $PBS_NODEFILE); do
  ssh $h "for p in \$(pgrep -x replace_tpdp); do gdb -p \$p -batch -ex 'thread apply all bt' > $OUT/stack.\$(hostname -s).\$p 2>&1; done"
done
echo "stacks taken; waiting for the run to end (max 300 s)"
wait $JOB
echo "== $NET victim $V $C deadline $DL $K rc=$? $OUT"
{ grep -E '^(pos|spare) |\.(cpp|h):[0-9]+ ' $OUT/run.log | sort -n -k2
  for f in $OUT/stack.*; do  # main thread's NCCL/CUDA frames, innermost first
    echo "${f##*/}: $(awk '/^Thread 1 /{f=1} f && /^#/' $f | grep -oE 'nccl[A-Za-z]+|cuda[A-Za-z]+|main' | uniq | head -4 | tr '\n' ' ')"
  done; } | tee $OUT/summary.txt
