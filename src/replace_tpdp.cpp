// TP+DP crash replace: 2 comms per rank on one stream; stop all, rebuild each, Grow the spare into
// the victim's two comms. Design: DECISIONS_2026_11_23a.
#include <algorithm>
#include <signal.h>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>
#include <nccl.h>
#include "store.h"

// Odd names inside the macros: a plain `r` would shadow a caller's `r` passed in x.
#define CUDA(x) do { cudaError_t cuda_err_ = x; if (cuda_err_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(cuda_err_)); exit(1); } } while (0)
#define NCCL(x) do { ncclResult_t nccl_res_ = x; if (nccl_res_ != ncclSuccess && nccl_res_ != ncclInProgress) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(nccl_res_)); exit(1); } } while (0)
#define DIE(...) do { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); exit(1); } while (0)

using clk = std::chrono::steady_clock;
using sec = std::chrono::duration<double>;
static const int kSteps = 25;           // x2 ops: same op count as 22d
static const size_t kCount = 64 << 20;  // 256 MiB of float
static const double kGrace = 3;

static int env(const char* name) {
  const char* v = getenv(name);
  if (!v) DIE("%s not set (run under PALS mpiexec)", name);
  return atoi(v);
}
static std::string bytes(const ncclUniqueId& id) { return std::string((const char*)&id, sizeof id); }
static ncclUniqueId toId(const std::string& s) {
  ncclUniqueId id;
  if (s.size() != sizeof id) DIE("bad id size %zu", s.size());
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
static int idx(const std::vector<int>& v, int x) { return int(std::find(v.begin(), v.end(), x) - v.begin()); }

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

struct Comm {
  std::string name;      // tp<node> or dp<local>
  std::vector<int> mem;  // positions in rank order
  ncclComm_t c = nullptr;
  bool hit = false;
};

int main(int argc, char** argv) {
  if (argc < 2) DIE("usage: %s <fresh run dir> [victim position] [revoke|abort] [deadline s]", argv[0]);
  Store st{argv[1]};
  int prank = env("PALS_RANKID"), n = env("PMI_SIZE"), local = env("PALS_LOCAL_RANKID"), L = env("PALS_LOCAL_SIZE");
  int T = L - 1, N = n / L, nAct = N * T, node = prank / L;  // assumes --ppn block placement
  int victim = argc > 2 ? atoi(argv[2]) : 1;
  bool revoke = !(argc > 3 && !strcmp(argv[3], "abort"));
  double deadline = argc > 4 ? atof(argv[4]) : 10;
  if (victim < 0 || victim >= nAct) DIE("victim must be in [0, %d)", nAct);
  bool isSpare = local == T;
  int pos = isSpare ? -1 : node * T + local;
  CUDA(cudaSetDevice(local));  // before any NCCL init
  cudaStream_t s;
  CUDA(cudaStreamCreate(&s));  // one stream for both comms
  int* d;
  CUDA(cudaMalloc(&d, (nAct + 1) * sizeof(int)));
  float* buf;
  CUDA(cudaMalloc(&buf, kCount * sizeof(float)));
  CUDA(cudaMemset(buf, 0, kCount * sizeof(float)));
  char host[64];
  gethostname(host, sizeof host);

  std::string times;
  auto t = clk::now();
  auto lap = [&](const std::string& name) {
    char b[64];
    snprintf(b, sizeof b, " %s %.3f", name.c_str(), sec(clk::now() - t).count());
    times += b;
    t = clk::now();
  };
  // Nonblocking: Shrink/Split take the parent's blocking mode (DECISIONS_2026_11_22c).
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
  cfg.blocking = 0;
  auto initComm = [&](const std::string& key, const std::vector<int>& mem) {
    ncclUniqueId id;
    if (mem[0] == pos) {
      NCCL(ncclGetUniqueId(&id));
      st.put(key, bytes(id));
    } else {
      if (!st.wait(key, 60)) DIE("pos %d: no %s after 60 s", pos, key.c_str());
      id = toId(st.get(key));
    }
    ncclComm_t c;
    NCCL(ncclCommInitRankConfig(&c, int(mem.size()), id, idx(mem, pos), &cfg));
    ready(c);
    return c;
  };

  Comm tp, dp;
  std::vector<Comm*> comms{&tp, &dp};  // fixed order on every rank: TP first, then DP
  bool ok = true;

  if (!isSpare) {
    st.put("host." + std::to_string(pos), host);
    tp.name = "tp" + std::to_string(node);
    dp.name = "dp" + std::to_string(local);
    for (int j = 0; j < T; j++) tp.mem.push_back(node * T + j);
    for (int k = 0; k < N; k++) dp.mem.push_back(k * T + local);
    for (Comm* c : comms) c->c = initComm(c->name + ".id", c->mem);
    lap("init");
    for (Comm* c : comms) ok &= check(c->c, pos, c->mem, d, s);
    lap("coll");
    auto enq = [&](Comm& c) {  // errors are left to the detect loop
      ncclResult_t a = ncclAllReduce(buf, buf, kCount, ncclFloat, ncclSum, c.c, s);
      while (a == ncclInProgress) NCCL(ncclCommGetAsyncError(c.c, &a));
      return a == ncclSuccess;
    };
    auto batch = [&](int steps) {
      for (int i = 0; i < steps; i++)
        if (!enq(tp) || !enq(dp)) return;
    };
    batch(kSteps);  // warm-up = the crash ops, so nothing connects lazily after the crash
    CUDA(cudaStreamSynchronize(s));
    lap("warm");

    if (pos == victim) {
      batch(kSteps / 2);  // half: peers' later ops can never finish
      usleep(20000);
      printf("pos %d victim kill%s\n", pos, times.c_str());
      fflush(stdout);
      kill(getpid(), SIGKILL);
    }

    // Detect: async error on either comm, a peer's notice, or the deadline.
    batch(kSteps);
    auto t0 = clk::now();
    auto since = [&] { return sec(clk::now() - t0).count(); };
    double tAsync = -1, nextStore = 0;
    std::string asyncOn = "-";
    auto pollAsync = [&] {
      for (Comm* c : comms) {
        ncclResult_t a;
        NCCL(ncclCommGetAsyncError(c->c, &a));
        if (a != ncclSuccess && a != ncclInProgress && tAsync < 0) { tAsync = since(); asyncOn = c->name; }
      }
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
    for (double tDet = since(); since() < tDet + kGrace;) {
      pollAsync();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    lap("grace");
    char b[96];
    snprintf(b, sizeof b, " by %s async %.3f on %s", by.c_str(), tAsync, asyncOn.c_str());
    times += b;

    // Agree: missing alive files = exclude; first writer wins.
    std::vector<int> ex;
    for (int p = 0; p < nAct; p++)
      if (!st.wait("alive." + std::to_string(p), 0)) ex.push_back(p);
    st.putOnce("exclude", join(ex));
    if (!st.wait("exclude", 10)) DIE("pos %d: no exclude", pos);
    ex = ints(st.get("exclude"));
    bool fenced = std::count(ex.begin(), ex.end(), pos);
    if (!fenced && ex.size() != 1) DIE("pos %d: exclude '%s', want one rank", pos, join(ex).c_str());
    lap("agree");

    // Stop all comms before draining any: draining one waits on the other's kernels on our stream.
    std::vector<std::string> why(comms.size());
    if (revoke) {
      for (size_t i = 0; i < comms.size(); i++) {
        ncclResult_t res = ncclCommRevoke(comms[i]->c, NCCL_REVOKE_DEFAULT);
        if (res != ncclSuccess && res != ncclInProgress) why[i] = ncclGetErrorString(res);
      }
    } else {  // Abort drains before it returns, even nonblocking (group.cc:56): one thread per comm
      std::vector<std::thread> th;
      for (Comm* c : comms) th.emplace_back([c] { ncclCommAbort(c->c); c->c = nullptr; });
      for (auto& x : th) x.join();
    }
    for (size_t i = 0; revoke && i < comms.size(); i++) {
      ncclResult_t a = ncclInProgress;
      while (why[i].empty() && a == ncclInProgress) NCCL(ncclCommGetAsyncError(comms[i]->c, &a));
      if (why[i].empty() && a != ncclSuccess) why[i] = ncclGetErrorString(a);  // e.g. sticky proxy error
    }
    lap(revoke ? "revoke" : "abort");
    CUDA(cudaStreamSynchronize(s));
    lap("drain");
    if (fenced) {  // too slow to write alive: the others dropped us
      printf("pos %d fenced%s\n", pos, times.c_str());
      return 0;
    }
    victim = ex[0];

    // Rebuild each comm: Shrink (hit) or Split (clean) of the revoked comm; any member fails → all re-init.
    for (size_t i = 0; i < comms.size(); i++) {
      Comm& c = *comms[i];
      std::string k = c.name + ".";
      int vr = idx(c.mem, victim), myOld = idx(c.mem, pos);
      c.hit = vr < int(c.mem.size());
      if (c.hit) c.mem.erase(c.mem.begin() + vr);
      ncclComm_t child = nullptr;
      bool all = false;
      if (revoke) {
        if (why[i].empty()) {
          ncclResult_t res = c.hit ? ncclCommShrink(c.c, &vr, 1, &child, &cfg, NCCL_SHRINK_DEFAULT)
                                   : ncclCommSplit(c.c, 0, myOld, &child, &cfg);
          if (res != ncclSuccess && res != ncclInProgress) why[i] = ncclGetErrorString(res);
          // child stays NULL until its init ends; progress shows on the parent (22c run 2).
          for (auto t1 = clk::now(); why[i].empty();) {
            ncclResult_t a;
            NCCL(ncclCommGetAsyncError(c.c, &a));
            if (a == ncclSuccess && child) break;
            if (a != ncclSuccess && a != ncclInProgress) why[i] = ncclGetErrorString(a);
            else if (st.wait(k + "fail", 0)) why[i] = "peer";
            else if (clk::now() - t1 > sec(60)) why[i] = "timeout";
            else std::this_thread::sleep_for(std::chrono::milliseconds(10));
          }
        }
        if (!why[i].empty()) {
          if (why[i] != "peer") st.put(k + "fail", std::to_string(pos));
          NCCL(ncclCommAbort(c.c));  // also aborts the child's init
          c.c = nullptr;
          if (child) NCCL(ncclCommAbort(child));
          child = nullptr;
        }
        lap(k + (c.hit ? "shrink" : "split"));
        // Outcomes are final once written, so all members pick the same path.
        st.put(k + "built." + std::to_string(pos), why[i].empty() ? "ok" : why[i]);
        all = true;
        for (int p : c.mem) {
          std::string key = k + "built." + std::to_string(p);
          if (!st.wait(key, 120)) DIE("pos %d: no %s", pos, key.c_str());
          all &= st.get(key) == "ok";
        }
        lap(k + "barrier");
        if (!all && child) NCCL(ncclCommAbort(child));
        if (c.c) NCCL(ncclCommAbort(c.c));  // the revoked parent
        c.c = all ? child : nullptr;
      }
      if (!all) {
        c.c = initComm(k + "id2", c.mem);
        lap(k + "reinit");
      }
      ok &= check(c.c, pos, c.mem, d, s);
      lap(k + "coll");
    }

    // Grow the victim's comms. Coordinator = rank 0; map = new order (spare last) + who left and who joined.
    std::string vhost = st.get("host." + std::to_string(victim));
    for (Comm* c : comms) {
      if (!c->hit) continue;
      std::string k = c->name + ".";
      if (c->mem[0] == pos) {
        ncclUniqueId gid;
        NCCL(ncclCommGetUniqueId(c->c, &gid));
        ready(c->c);
        char m[256];
        snprintf(m, sizeof m, "%s%d | excluded pos %d host %s gpu %d | spare PALS %d host %s gpu %d",
                 join(c->mem).c_str(), victim, victim, vhost.c_str(), victim % T, victim / T * L + T, vhost.c_str(), T);
        st.put(k + "map", m);  // before grow_id: readers wait on grow_id
        st.put(k + "grow_id", bytes(gid));
      }
      ncclComm_t g;
      NCCL(ncclCommGrow(c->c, int(c->mem.size()) + 1, nullptr, -1, &g, nullptr));
      ready(g);
      NCCL(ncclCommDestroy(c->c));  // nonblocking: freed async, never touch it again
      c->c = g;
      lap(k + "grow");
    }
  } else {
    if (!st.wait("exclude", deadline + 120)) DIE("spare PALS %d: no exclude", prank);
    std::vector<int> ex = ints(st.get("exclude"));
    if (ex.size() != 1 || ex[0] / T != node) { printf("spare PALS %d unused\n", prank); return 0; }
    lap("wait");
    pos = victim = ex[0];
    tp.name = "tp" + std::to_string(node);
    dp.name = "dp" + std::to_string(victim % T);
    tp.hit = dp.hit = true;
  }

  // Everyone in a grown comm reads the map and reports the reorder; the spare joins here.
  for (Comm* c : comms) {
    if (!c->hit) continue;
    std::string k = c->name + ".";
    if (!st.wait(k + "grow_id", 120)) DIE("pos %d: no %sgrow_id", pos, k.c_str());
    std::string m = st.get(k + "map");
    c->mem = ints(m);
    printf("%s_REORDER comm=%s pos %d rank %d: order(pos) %s\n", c->name[0] == 't' ? "TP" : "DP", c->name.c_str(), pos,
           idx(c->mem, pos), m.c_str());
    if (isSpare) {
      ncclUniqueId gid = toId(st.get(k + "grow_id"));
      int sz = int(c->mem.size());
      NCCL(ncclCommGrow(nullptr, sz, &gid, sz - 1, &c->c, nullptr));
      lap(k + "grow");
    }
  }
  for (Comm* c : comms) ok &= check(c->c, pos, c->mem, d, s);
  lap("collg");
  printf("%s %d PALS %d %s %s%s %s\n", isSpare ? "spare" : "pos", pos, prank, tp.name.c_str(), dp.name.c_str(),
         times.c_str(), ok ? "PASS" : "FAIL");
  fflush(stdout);
  for (Comm* c : comms) {
    NCCL(ncclCommFinalize(c->c));
    ready(c->c);
    NCCL(ncclCommDestroy(c->c));
  }
  CUDA(cudaFree(buf));
  CUDA(cudaFree(d));
  return !ok;
}
