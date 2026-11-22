// Crash replace: the victim is SIGKILLed (or hangs) mid-batch; survivors detect, agree via the store,
// Shrink (or re-init), then Grow with a same-node spare. Design: DECISIONS_2026_11_22c.
#include <algorithm>
#include <numeric>
#include <signal.h>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>
#include <nccl.h>
#include "store.h"

#define CUDA(x) do { cudaError_t e = x; if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
// InProgress is not an error: nonblocking calls are settled by ready().
#define NCCL(x) do { ncclResult_t r = x; if (r != ncclSuccess && r != ncclInProgress) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); exit(1); } } while (0)

using clk = std::chrono::steady_clock;
using sec = std::chrono::duration<double>;
static const int kIters = 50;
static const size_t kCount = 64 << 20;  // 256 MiB of float
static const double kGrace = 3;

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

static std::string join(const std::vector<int>& v) {
  std::string s;
  for (int x : v) s += std::to_string(x) + " ";
  return s;
}
static std::vector<int> ints(const std::string& s) {
  std::istringstream in(s);
  std::vector<int> v;
  for (int x; in >> x;) v.push_back(x);
  return v;
}

static void ready(ncclComm_t c) {
  ncclResult_t a;
  do { NCCL(ncclCommGetAsyncError(c, &a)); } while (a == ncclInProgress);
  NCCL(a);
}

// AllGather each rank's position; want[r] = position expected at rank r.
static bool check(ncclComm_t comm, int pos, const std::vector<int>& want, int* d, cudaStream_t s) {
  std::vector<int> got(want.size());
  CUDA(cudaMemcpy(d, &pos, sizeof pos, cudaMemcpyHostToDevice));
  NCCL(ncclAllGather(d, d + 1, 1, ncclInt, comm, s));
  ready(comm);
  CUDA(cudaStreamSynchronize(s));
  CUDA(cudaMemcpy(got.data(), d + 1, got.size() * sizeof(int), cudaMemcpyDeviceToHost));
  return got == want;
}

// map file: "spare <PALS rank>", then "pos <p> rank <r>" lines. Returns the spare; fills want (rank → pos).
static int readMap(const Store& st, std::vector<int>& want, double timeoutS) {
  if (!st.wait("grow_id", timeoutS)) { fprintf(stderr, "no grow_id after %.0f s\n", timeoutS); exit(1); }
  std::istringstream in(st.get("map"));
  std::string w;
  int spare, p, r;
  in >> w >> spare;
  while (in >> w >> p >> w >> r) want.at(r) = p;
  return spare;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <fresh run dir> [victim position] [kill|hang] [deadline s]\n", argv[0]); return 1; }
  Store st{argv[1]};
  int prank = env("PALS_RANKID"), n = env("PMI_SIZE"), local = env("PALS_LOCAL_RANKID"), L = env("PALS_LOCAL_SIZE");
  int nAct = n - n / L, victim = argc > 2 ? atoi(argv[2]) : 1;
  bool hang = argc > 3 && !strcmp(argv[3], "hang");
  double deadline = argc > 4 ? atof(argv[4]) : 10;
  if (victim < 0 || victim >= nAct) { fprintf(stderr, "victim must be in [0, %d)\n", nAct); return 1; }
  bool isSpare = local == L - 1;
  int pos = isSpare ? -1 : prank / L * (L - 1) + local;  // assumes --ppn block placement
  CUDA(cudaSetDevice(local));  // before any NCCL init
  cudaStream_t s;
  CUDA(cudaStreamCreate(&s));
  int* d;
  CUDA(cudaMalloc(&d, (nAct + 1) * sizeof(int)));
  float* buf;
  CUDA(cudaMalloc(&buf, kCount * sizeof(float)));
  CUDA(cudaMemset(buf, 0, kCount * sizeof(float)));

  std::string times;
  auto t = clk::now();
  auto lap = [&](const char* name) {
    char b[48];
    snprintf(b, sizeof b, " %s %.3f", name, sec(clk::now() - t).count());
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
    // Nonblocking: Shrink takes its blocking mode from the parent, not the child config (DECISIONS_2026_11_22c).
    ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
    cfg.blocking = 0;
    ncclComm_t old;
    NCCL(ncclCommInitRankConfig(&old, nAct, id, pos, &cfg));
    ready(old);
    lap("init");
    std::vector<int> cur(nAct);
    std::iota(cur.begin(), cur.end(), 0);
    ok &= check(old, pos, cur, d, s);
    lap("coll6");
    auto batch = [&] {  // errors are left to the detect loop: the victim may die mid-enqueue
      for (int i = 0; i < kIters; i++) {
        ncclResult_t a = ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, old, s);
        while (a == ncclInProgress) NCCL(ncclCommGetAsyncError(old, &a));  // next call fails while InProgress
        if (a != ncclSuccess) break;
      }
    };
    batch();  // warm-up = the crash op, so nothing connects lazily after the crash
    CUDA(cudaStreamSynchronize(s));
    lap("warm");

    if (pos == victim) {
      if (!hang) {
        batch();
        usleep(20000);
        printf("pos %d victim kill%s\n", pos, times.c_str());
        fflush(stdout);
        kill(getpid(), SIGKILL);
      }
      printf("pos %d victim hang%s\n", pos, times.c_str());
      fflush(stdout);
      sleep(unsigned(deadline + 2 * kGrace));  // a stuck host thread: no batch, no alive file
      std::vector<int> ex = st.wait("exclude", 60) ? ints(st.get("exclude")) : std::vector<int>{};
      bool fenced = std::count(ex.begin(), ex.end(), pos);
      printf("pos %d victim woke %s\n", pos, fenced ? "fenced" : "NOT fenced");
      NCCL(ncclCommAbort(old));
      return !fenced;
    }

    // Detect: async error, a peer's notice, or the deadline.
    batch();
    auto t0 = clk::now();
    auto since = [&] { return sec(clk::now() - t0).count(); };
    double tAsync = -1, nextStore = 0;
    auto pollAsync = [&] {
      ncclResult_t a;
      NCCL(ncclCommGetAsyncError(old, &a));
      if (a != ncclSuccess && a != ncclInProgress && tAsync < 0) tAsync = since();
    };
    std::string by;
    while (by.empty()) {
      cudaError_t q = cudaStreamQuery(s);
      if (q == cudaSuccess) { printf("pos %d batch finished, no failure seen%s\n", pos, times.c_str()); return 1; }
      if (q != cudaErrorNotReady) CUDA(q);  // local fatal error, not a peer failure
      pollAsync();
      if (tAsync >= 0) by = "async";
      else if (since() > deadline) by = "deadline";
      else if (since() > nextStore) { nextStore = since() + 0.01; if (st.wait("fail", 0)) by = "notice"; }
      else std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    lap("detect");
    if (by != "notice") st.put("fail", by);
    st.put("alive." + std::to_string(pos), "");
    for (double tDet = since(); since() < tDet + kGrace;) {  // async stays polled: measures time-to-error
      pollAsync();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    lap("grace");
    char b[64];
    snprintf(b, sizeof b, " by %s async %.3f", by.c_str(), tAsync);
    times += b;

    // Agree: missing alive files = exclude; first writer wins.
    std::vector<int> ex;
    for (int p = 0; p < nAct; p++)
      if (!st.wait("alive." + std::to_string(p), 0)) ex.push_back(p);
    st.putOnce("exclude", join(ex));
    if (!st.wait("exclude", 10)) { fprintf(stderr, "pos %d: no exclude\n", pos); return 1; }
    ex = ints(st.get("exclude"));
    if (std::count(ex.begin(), ex.end(), pos)) {  // too slow to write alive: the others dropped us
      printf("pos %d fenced%s\n", pos, times.c_str());
      NCCL(ncclCommAbort(old));
      return 0;
    }
    if (ex.size() != 1) { fprintf(stderr, "pos %d: exclude '%s', want one rank\n", pos, join(ex).c_str()); return 1; }
    victim = ex[0];
    cur.erase(cur.begin() + victim);
    lap("agree");

    // Nonblocking Shrink; if any survivor fails (sticky async error), all re-init. See DECISIONS_2026_11_22c.
    ncclComm_t shr = nullptr;
    ncclResult_t sr = ncclCommShrink(old, &victim, 1, &shr, &cfg, NCCL_SHRINK_ABORT);
    bool shrunk = sr == ncclSuccess || sr == ncclInProgress;
    std::string why = shrunk ? "" : ncclGetErrorString(sr);
    for (auto t1 = clk::now(); shrunk;) {
      ncclResult_t a;
      NCCL(ncclCommGetAsyncError(shr, &a));
      if (a == ncclSuccess) break;
      if (a != ncclInProgress) why = ncclGetErrorString(a);
      else if (st.wait("shrinkfail", 0)) why = "peer";
      else if (clk::now() - t1 > sec(60)) why = "timeout";
      else { std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue; }
      shrunk = false;
    }
    if (!shrunk && why != "peer") st.put("shrinkfail", std::to_string(pos));
    if (!shrunk && shr) NCCL(ncclCommAbort(shr));
    lap(shrunk ? "shrink" : "shrinkfail");
    // Outcomes are final once written, so all survivors pick the same path.
    st.put("shrunk." + std::to_string(pos), shrunk ? "ok" : why);
    bool all = true;
    for (int p : cur) {
      std::string k = "shrunk." + std::to_string(p);
      if (!st.wait(k, 120)) { fprintf(stderr, "pos %d: no %s\n", pos, k.c_str()); return 1; }
      all &= st.get(k) == "ok";
    }
    lap("barrier");
    if (shrunk && !all) NCCL(ncclCommAbort(shr));
    NCCL(ncclCommAbort(old));
    lap("abort");
    CUDA(cudaStreamSynchronize(s));
    lap("drain");
    if (!all) {
      ncclUniqueId id5;
      if (cur[0] == pos) {
        NCCL(ncclGetUniqueId(&id5));
        st.put("id5", bytes(id5));
      } else {
        if (!st.wait("id5", 60)) { fprintf(stderr, "pos %d: no id5 after 60 s\n", pos); return 1; }
        id5 = toId(st.get("id5"));
      }
      NCCL(ncclCommInitRank(&shr, nAct - 1, id5, int(std::find(cur.begin(), cur.end(), pos) - cur.begin())));
      lap("reinit");
    }
    times += all ? " path shrink" : " path reinit";

    ok &= check(shr, pos, cur, d, s);
    lap("coll5");
    int r5;
    NCCL(ncclCommUserRank(shr, &r5));
    if (r5 == 0) {  // coordinator
      ncclUniqueId gid;
      NCCL(ncclCommGetUniqueId(shr, &gid));
      ready(shr);
      cur.push_back(victim);  // Grow appends: the spare takes the last rank
      std::string m = "spare " + std::to_string(victim / (L - 1) * L + L - 1) + "\n";
      for (int p = 0; p < nAct; p++)
        m += "pos " + std::to_string(p) + " rank " + std::to_string(std::find(cur.begin(), cur.end(), p) - cur.begin()) + "\n";
      st.put("map", m);  // before grow_id: readers wait on grow_id
      st.put("grow_id", bytes(gid));
      lap("getid");
    }
    NCCL(ncclCommGrow(shr, nAct, nullptr, -1, &grown, nullptr));
    ready(grown);
    lap("grow");
    NCCL(ncclCommDestroy(shr));  // nonblocking: freed async, never touch shr again
    readMap(st, want, 60);
    snprintf(who, sizeof who, "pos %d rank %d->%d->", pos, pos, r5);
  } else {
    int spare = readMap(st, want, deadline + 120);
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
  NCCL(ncclCommFinalize(grown));  // survivors' grown comm inherits nonblocking from shr
  ready(grown);
  NCCL(ncclCommDestroy(grown));
  CUDA(cudaFree(buf));
  CUDA(cudaFree(d));
  return !ok;
}
