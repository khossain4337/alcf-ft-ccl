# alcf-ft-ccl: replace a failed GPU in a running NCCL job

A demo of the NCCL fault-tolerance API (`ncclCommRevoke`, `ncclCommShrink`,
`ncclCommGrow`, `ncclCommAbort`) on Tara (2 nodes x 4 GPUs, NCCL 2.32.6, CUDA 13.0).
One rank dies. The survivors detect it, agree on who is gone, drop it, and pull
in an idle spare GPU. The job keeps running, with no restart.
What oneCCL would need for the same: `oneccl_minimal_requirements.md`.

NCCL calls we needed beyond a plain init + collectives:
| need | call |
|---|---|
| nonblocking comms (Shrink/Revoke progress needs them) | `ncclCommInitRankConfig`, `blocking = 0` |
| detect, and settle nonblocking calls | `ncclCommGetAsyncError` |
| stop a broken comm | `ncclCommRevoke` (`NCCL_REVOKE_DEFAULT`) or `ncclCommAbort` |
| drop the dead rank | `ncclCommShrink`: `NCCL_SHRINK_ABORT` on a live comm, `NCCL_SHRINK_DEFAULT` after Revoke |
| rebuild comms without the victim | `ncclCommSplit` after Revoke, or re-init from a fresh id |
| add the spare | `ncclCommGetUniqueId` on the shrunk comm, then `ncclCommGrow` on all |

## Setup
- One `mpiexec -n 8 --ppn 4` launch with PALS `--no-abort-on-failure`, so the job
  survives a dead rank. 6 active ranks, 1 idle spare per node.
- No MPI inside the app. Ranks swap NCCL ids and failure notices through a small
  file store on `/vast` (`src/store.h`). This is for the demo only; at scale a TCP store would replace it.
- Networks: Socket (TCP) and OFI (Slingshot/CXI via aws-ofi-nccl).

## Recovery steps
1. Detect: a no-progress deadline, `ncclCommGetAsyncError`, or a notice in the store from another rank.
2. Agree: each survivor writes `alive.<pos>`. After a grace period, the first one writes the exclude list.
3. Stop: Revoke or Abort the comms.
4. Rebuild: Shrink out the dead rank (or re-init from a fresh id), then Grow the spare in.
5. Check: confirm every comm has the right members in the right order (below).

## Stages
| stage | source | what it shows |
|---|---|---|
| cooperative | `replace_cooperative.cpp` | Victim leaves cleanly; Shrink + Grow. No failure. |
| crash | `replace_crash.cpp` | Victim dies mid-collective; one comm. |
| TP+DP | `replace_tpdp.cpp` | TP 3 per node (NVLink), DP 2 across nodes; 5 comms share one stream per rank. |

## Terms
- `pos`: a rank's logical place in the job (0-5). The spare takes over the victim's `pos`.
- `kill` / `hang`: the victim gets `SIGKILL`, or its host thread stops while the process stays up.
- `revoke` path: Revoke all comms. Comms that held the victim are Shrunk; the others are Split. Then Grow.
- `abort` path: Abort all comms, one thread per comm. Re-init every comm from a fresh id. Then Grow.
- `mainthread`: one thread runs the training loop. It enqueues a short batch, then watches it.
  The same thread that issued the collectives also detects and recovers. This is the simple case.
- `fullqueue`: a training loop with a watchdog, as in PyTorch. A feeder thread enqueues
  collectives far ahead until the CUDA launch queue fills and the feeder blocks. A second
  thread acts as the watchdog. It detects from host-side counters only, then stops all comms
  while the feeder may still be inside NCCL.
- `feederIn 1|0`: was the feeder still inside a NCCL call when the stop began?
  Only `1` exercises the dangerous timing (the use-after-free in Findings).

## Status (victim pos 1 unless noted)
Cooperative: PASS for victims 0, 1, 5 on Socket and OFI.

Crash (`Shrink(ABORT)` + `Abort(old)`; on any failure, all ranks re-init):
| case | Socket | OFI |
|---|---|---|
| kill | PASS (victims 0, 1, 5) | PASS |
| hang | PASS | not run |

TP+DP:
| case | path | Socket | OFI |
|---|---|---|---|
| `mainthread` | revoke | PASS | PASS |
| `mainthread` | abort | PASS | PASS |
| `fullqueue` | revoke | hang, not localized | PASS x3, `feederIn 0` |
| `fullqueue` | abort | no result yet | PASS x3, `feederIn 0` |

The `fullqueue` OFI passes cover only the common timing (`feederIn 0`).

Goal: both Revoke and Abort PASS under a realistic workload (`fullqueue`), on both networks,
including `feederIn 1`. `mainthread` is a stepping stone.
- Revoke: works from one thread and keeps clean comms (Split). PyTorch uses it when it reconfigures.
- Abort: NCCL's documented way to stop a comm from another thread. PyTorch falls back to it.
- Frameworks take either path, so a replace design must cover both.

## Correctness checks
- After init, after each rebuild, and after Grow, every comm runs an AllGather of each rank's `pos`.
- The gathered list must match the expected member order exactly. Any mismatch prints `FAIL`.
- Before a rebuilt comm is used, every member must report `<comm>.built.<pos>: ok` in the store.
  If any member fails, all members re-init that comm.
- The exclude list must name only the victim.
- Grow appends the spare as the last rank, so the rank order can change (`TP_REORDER`).
  The run prints it loudly: the app must map `pos` to rank, not assume they are equal.
- A run passes when all 5 survivors and the used spare print `PASS`, and the other spare prints `unused`.
- Not checked: the numeric result of the AllReduce traffic.

## Findings
- Several comms share one stream: stop them all before draining any.
- `ncclCommAbort` always drains in the caller, even on a nonblocking comm (`group.cc:56`).
- So Abort needs one thread per comm. Revoke works from one thread.
- NCCL 2.32.6 bug: Revoke or Abort while another thread is in `ncclCommGetAsyncError` on that comm.
  Result: use-after-free of the group job (`init.cc:4104`, `group.cc:1198`).
- PyTorch's `nccl2` watchdog has the same pattern.
- A full CUDA launch queue blocks CUDA calls on that stream from other threads.
- So a watchdog must detect without CUDA calls on that stream.
- OFI/CXI gave no async error in 60 s. On Slingshot only the deadline detects.
- Socket: only the victim's network neighbour gets a sticky error. That comm cannot Shrink; it re-inits.
- At 6 ranks the first collective on a new comm (1-5 s) dominates recovery.
- Shrink (0.16-1 s) is no faster than a warm re-init (<0.1 s) at this scale.

## Open
- Socket `fullqueue`: localize the revoke hang; rerun abort.
- Force `feederIn 1` to test the watchdog's bounded wait.
- Does `cudaEventQuery` block on a full launch queue? The PyTorch watchdog relies on it.

## Running on Tara
- Build: `scripts/build_<stage>_tara.sh`. Paths and env: `scripts/functionality/env_tara.sh`.
- `run_replace_cooperative_tara.sh [victim] [nets]`
- `run_replace_crash_tara.sh [victim] [kill|hang] [deadline] [nets]`
- `run_replace_tpdp_tara.sh [victim] [revoke|abort] [deadline] [mainthread|fullqueue] [nets]`
- `stacks_replace_tpdp_tara.sh [victim] [revoke|abort] [deadline] [net] [case] [at s]`: one run plus gdb stacks of all ranks.
- Logs and `summary.txt` go to `$SOFT_ROOT/testing/<stage>/`.
- Design notes and run records live outside this repo; ask Khalid.

## Typical output
TP+DP, OFI, `fullqueue revoke`, victim pos 1 (trimmed; times in seconds):
```
pos 1 victim kill init 2.999 coll 7.450 warm 1.103
... rank 1 died from signal 9
spare PALS 7 unused
pos 4 PALS 5 tp1 dp1 init 3.071 coll 7.467 warm 1.103 detect 30.376 grace 3.009 by deadline
      async -1.000 on - queued 59 agree 0.010 feederIn 0 revoke 1.749 drain 0.000
      tp1.split 0.198 tp1.barrier 0.017 tp1.coll 2.539 dp1.shrink 0.131 dp1.barrier 0.006
      dp1.coll 0.018 dp1.grow 6.979 collg 5.429 PASS
spare 1 PALS 3 tp0 dp1 wait 44.948 tp0.grow 10.997 dp1.grow 0.072 collg 5.996 PASS
TP_REORDER comm=tp0 pos 0 rank 0: order(pos) 0 2 1 | excluded pos 1 host x4820c7s6b1n0 gpu 1 | spare PALS 3 host x4820c7s6b1n0 gpu 3
```
| field | meaning |
|---|---|
| `pos 4 PALS 5 tp1 dp1` | logical position, launcher rank, the two comms it belongs to |
| `init`, `coll`, `warm` | first init, first collective, one warm collective |
| `detect ... by` | time to detect, and how: `deadline`, `notice` (store), or `async` (NCCL error) |
| `queued` | collectives enqueued but not done at detection |
| `agree` | time to agree on the exclude list |
| `revoke` / `abort`, `drain` | time to stop all comms, then to drain the stream |
| `<comm>.shrink/.split/.reinit/.grow` | rebuild time per comm |
| `<comm>.coll`, `collg` | first check after the rebuild, and after Grow |
| `TP_REORDER ... order(pos) 0 2 1` | new rank order of `tp0`: the spare (pos 1) is now rank 2 |

Here pos 4 detected by its own deadline; the others got the store notice from it.
`dp1` (the victim's DP comm) shrank to 1 rank and grew back to 2.
