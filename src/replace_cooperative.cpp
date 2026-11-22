// Cooperative replace: one known victim leaves; survivors Shrink, then Grow with a same-node spare.
// Design: DECISIONS_2026_11_22b. Last local rank on each node is a spare.
#include <algorithm>
#include <numeric>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>
#include <nccl.h>
#include "store.h"

#define CUDA(x) do { cudaError_t e = x; if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define NCCL(x) do { ncclResult_t r = x; if (r != ncclSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); exit(1); } } while (0)

using clk = std::chrono::steady_clock;

static int env(const char* name) {
  const char* v = getenv(name);
  if (!v) { fprintf(stderr, "%s not set (run under PALS mpiexec)\n", name); exit(1); }
  return atoi(v);
}

static std::string bytes(const ncclUniqueId& id) { return std::string((const char*)&id, sizeof id); }
static ncclUniqueId toId(const std::string& s) {
  ncclUniqueId id;
  if (s.size() != sizeof id) { fprintf(stderr, "bad id size %zu\n", s.size()); exit(1); }
  memcpy(&id, s.data(), sizeof id);
  return id;
}

// AllGather each rank's position; want[r] = position expected at rank r.
static bool check(ncclComm_t comm, int pos, const std::vector<int>& want, int* d, cudaStream_t s) {
  std::vector<int> got(want.size());
  CUDA(cudaMemcpy(d, &pos, sizeof pos, cudaMemcpyHostToDevice));
  NCCL(ncclAllGather(d, d + 1, 1, ncclInt, comm, s));
  CUDA(cudaStreamSynchronize(s));
  CUDA(cudaMemcpy(got.data(), d + 1, got.size() * sizeof(int), cudaMemcpyDeviceToHost));
  return got == want;
}

// map file: "spare <PALS rank>", then "pos <p> rank <r>" lines. Returns the spare; fills want (rank → pos).
static int readMap(const Store& st, std::vector<int>& want) {
  if (!st.wait("grow_id", 120)) { fprintf(stderr, "no grow_id after 120 s\n"); exit(1); }
  std::istringstream in(st.get("map"));
  std::string w;
  int spare, p, r;
  in >> w >> spare;
  while (in >> w >> p >> w >> r) want.at(r) = p;
  return spare;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <fresh run dir> [victim position]\n", argv[0]); return 1; }
  Store st{argv[1]};
  int prank = env("PALS_RANKID"), n = env("PMI_SIZE"), local = env("PALS_LOCAL_RANKID"), L = env("PALS_LOCAL_SIZE");
  int nAct = n - n / L, victim = argc > 2 ? atoi(argv[2]) : 1;
  if (victim < 0 || victim >= nAct) { fprintf(stderr, "victim must be in [0, %d)\n", nAct); return 1; }
  bool isSpare = local == L - 1;
  int pos = isSpare ? -1 : prank / L * (L - 1) + local;  // assumes --ppn block placement
  CUDA(cudaSetDevice(local));  // before any NCCL init
  cudaStream_t s;
  CUDA(cudaStreamCreate(&s));
  int* d;
  CUDA(cudaMalloc(&d, (nAct + 1) * sizeof(int)));

  std::string times;
  auto t = clk::now();
  auto lap = [&](const char* name) {
    char b[48];
    snprintf(b, sizeof b, " %s %.3f", name, std::chrono::duration<double>(clk::now() - t).count());
    times += b;
    t = clk::now();
  };
  bool ok = true;
  std::vector<int> want(nAct, -1);
  ncclComm_t grown;
  char who[48];

  if (!isSpare) {
    ncclUniqueId id;
    if (prank == 0) {
      NCCL(ncclGetUniqueId(&id));
      st.put("id0", bytes(id));
    } else {
      if (!st.wait("id0", 60)) { fprintf(stderr, "pos %d: no id0 after 60 s\n", pos); return 1; }
      id = toId(st.get("id0"));
    }
    ncclComm_t old;
    NCCL(ncclCommInitRank(&old, nAct, id, pos));
    lap("init");
    std::vector<int> cur(nAct);
    std::iota(cur.begin(), cur.end(), 0);
    ok &= check(old, pos, cur, d, s);
    lap("coll6");

    if (pos == victim) {
      NCCL(ncclCommDestroy(old));  // survivors Destroy too, so its intra-node barrier completes
      printf("pos %d victim%s left %s\n", pos, times.c_str(), ok ? "PASS" : "FAIL");
      return !ok;
    }

    ncclComm_t shr;
    NCCL(ncclCommShrink(old, &victim, 1, &shr, nullptr, NCCL_SHRINK_DEFAULT));
    lap("shrink");
    NCCL(ncclCommDestroy(old));
    cur.erase(cur.begin() + victim);
    ok &= check(shr, pos, cur, d, s);
    lap("coll5");

    int r5;
    NCCL(ncclCommUserRank(shr, &r5));
    if (r5 == 0) {  // coordinator
      ncclUniqueId gid;
      NCCL(ncclCommGetUniqueId(shr, &gid));
      cur.push_back(victim);  // Grow appends: the spare takes the last rank
      std::string m = "spare " + std::to_string(victim / (L - 1) * L + L - 1) + "\n";
      for (int p = 0; p < nAct; p++)
        m += "pos " + std::to_string(p) + " rank " + std::to_string(std::find(cur.begin(), cur.end(), p) - cur.begin()) + "\n";
      st.put("map", m);  // before grow_id: readers wait on grow_id
      st.put("grow_id", bytes(gid));
      lap("getid");
    }
    NCCL(ncclCommGrow(shr, nAct, nullptr, -1, &grown, nullptr));
    lap("grow");
    NCCL(ncclCommDestroy(shr));
    readMap(st, want);
    snprintf(who, sizeof who, "pos %d rank %d->%d->", pos, pos, r5);
  } else {
    int spare = readMap(st, want);
    lap("wait");
    if (spare != prank) { printf("spare PALS %d unused\n", prank); return 0; }
    ncclUniqueId gid = toId(st.get("grow_id"));
    NCCL(ncclCommGrow(nullptr, nAct, &gid, nAct - 1, &grown, nullptr));
    lap("grow");
    pos = want[nAct - 1];
    snprintf(who, sizeof who, "pos %d spare PALS %d rank ", pos, prank);
  }

  ok &= check(grown, pos, want, d, s);
  lap("coll6");
  int rank;
  NCCL(ncclCommUserRank(grown, &rank));
  printf("%s%d%s %s\n", who, rank, times.c_str(), ok ? "PASS" : "FAIL");
  NCCL(ncclCommDestroy(grown));
  CUDA(cudaFree(d));
  return !ok;
}
