# oneCCL: minimal requirements for GPU replacement

What oneCCL must provide so an app can replace a failed GPU without a restart.
It comes from a working NCCL 2.32.6 demo on Tara (2 nodes x 4 GPUs, Socket and OFI/Slingshot);
see `README.md`. It is a starting point, not a final spec.

## Recovery sequence
1. Detect: a deadline, a library error, or a notice from another rank.
2. Agree: survivors agree on the failed ranks.
3. Stop: stop every comm that shares the stream (Revoke or Abort).
4. Rebuild: Shrink out the failed ranks, Split the clean comms, or re-init.
5. Grow: the spare joins; every comm is full size again.
6. Resume: the app maps its work to the new rank numbers.

## Required calls
| NCCL call | what it must do | oneCCL master (2026-11-03) |
|---|---|---|
| nonblocking mode | Init, Shrink, Split, Grow return at once; the app polls the state | `blocking` flag only; no polling state |
| `ncclCommGetAsyncError` | Return the comm state: OK, in progress, or error | declared; no output argument |
| `ncclCommAbort` | Stop and free a comm; never wait for a dead peer | declared, not implemented |
| `ncclCommRevoke` | Stop a comm, keep it for Shrink/Split | missing |
| `ncclCommShrink` | New comm without the listed ranks | missing |
| `ncclCommSplit` | New comm from a subset of a live comm | present |
| `ncclCommGetUniqueId(comm)` | Join id from a live comm, for the spare | missing |
| `ncclCommGrow` | Add ranks to a live comm | missing |

## Required behavior (each item broke our demo on NCCL)
1. Stopping a comm must be safe from another thread.
   NCCL 2.32.6 frees memory in use if one thread Revokes or Aborts while another is in `GetAsyncError`.
   A watchdog does exactly this.
2. `GetAsyncError` must never block. NCCL's can wait on a finished launch thread.
3. Stopping must not drain the stream. NCCL's Abort drains in the caller.
   With several comms on one stream, serial Aborts deadlock. We need one thread per comm.
4. Revoke must stop a comm from one thread with no wait. NCCL's does, unless a launch is stuck; then it waits too.
5. Shrink must work after a peer error. NCCL keeps the error forever and refuses Shrink. We re-init instead.
6. The call's own config must set blocking mode. NCCL's Shrink takes the parent's mode, so ranks blocked.
7. Say where to poll during Shrink. NCCL shows progress on the parent; the child handle is NULL until done.
8. Abort of the parent must also stop a child still being built. NCCL does this. It is the way out when one rank's Shrink fails.
9. Abort must return fast with a dead peer. NCCL: 0.05-0.8 s.
10. Grow must put new ranks last and keep the old order. The app keeps its own rank map.

## What the app must do (not the library)
- Detect a dead peer. OFI/CXI gave no error in 60 s; only a deadline works.
  Socket: only the dead rank's direct peer sees an error.
- Detect without CUDA calls on the stream. A full launch queue blocks them from other threads.
- Tell other ranks. Revoke and errors are local to one rank.
- Agree on the failed ranks. The demo uses files on a shared file system; at scale, a TCP store.
- Know which operations finished. Some ranks may finish a collective and others not. The app rolls back.

## Costs (6 ranks; rough)
| step | time |
|---|---|
| first init (cold process) | 2.2-2.7 s |
| stop all 5 comms (Revoke, or Abort one thread per comm) | 0.5-0.9 s |
| Shrink | 0.2-1 s |
| re-init (warm process) | <0.1 s |
| Grow | 0.7-1.3 s |
| first collective on a new comm | 0.7-5 s |
- At this size a warm re-init beats Shrink. Shrink may win at scale; not measured.
- Lazy connect makes the first collective slow. It dominates recovery.

## Not covered
- More than one failure. More than 2 nodes. Hangs inside the GPU or the network.
