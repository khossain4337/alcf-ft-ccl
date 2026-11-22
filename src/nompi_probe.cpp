#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <nccl.h>

#define CUDA(x) do { cudaError_t e = x; if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define NCCL(x) do { ncclResult_t r = x; if (r != ncclSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); exit(1); } } while (0)

using clk = std::chrono::steady_clock;
static double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

static int env(const char* name) {
  const char* v = getenv(name);
  if (!v) { fprintf(stderr, "%s not set (run under PALS mpiexec)\n", name); exit(1); }
  return atoi(v);
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <fresh run dir> [count]\n", argv[0]); return 1; }
  int rank = env("PALS_RANKID"), n = env("PMI_SIZE"), local = env("PALS_LOCAL_RANKID");
  std::string idPath = std::string(argv[1]) + "/id";
  size_t count = argc > 2 ? strtoull(argv[2], nullptr, 0) : 1 << 24;
  CUDA(cudaSetDevice(local));  // before ncclCommInitRank!

  auto t = clk::now();
  ncclUniqueId id;
  if (rank == 0) {
    NCCL(ncclGetUniqueId(&id));
    std::string tmp = idPath + ".tmp";  // rename is atomic: readers never see a partial id
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f || fwrite(&id, sizeof id, 1, f) != 1 || fclose(f) || rename(tmp.c_str(), idPath.c_str())) {
      perror(tmp.c_str());
      return 1;
    }
  } else {
    FILE* f;
    while (!(f = fopen(idPath.c_str(), "rb"))) {
      if (since(t) > 60) { fprintf(stderr, "rank %d: no %s after 60 s\n", rank, idPath.c_str()); return 1; }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (fread(&id, sizeof id, 1, f) != 1) { fprintf(stderr, "rank %d: short read\n", rank); return 1; }
    fclose(f);
  }
  double tId = since(t);

  t = clk::now();
  ncclComm_t comm;
  NCCL(ncclCommInitRank(&comm, n, id, rank));
  double tInit = since(t);

  size_t bytes = count * sizeof(float);
  std::vector<float> h(count, rank + 1.0f);
  float *send, *recv;
  CUDA(cudaMalloc(&send, bytes));
  CUDA(cudaMalloc(&recv, bytes));
  CUDA(cudaMemcpy(send, h.data(), bytes, cudaMemcpyHostToDevice));
  cudaStream_t s;
  CUDA(cudaStreamCreate(&s));
  t = clk::now();
  NCCL(ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, s));
  CUDA(cudaStreamSynchronize(s));
  double tAr = since(t);  // includes lazy connect
  CUDA(cudaMemcpy(h.data(), recv, bytes, cudaMemcpyDeviceToHost));

  float want = n * (n + 1) / 2.0f;
  unsigned long bad = 0;
  for (float v : h) bad += v != want;
  printf("rank %d/%d local %d id %.3f s init %.3f s allreduce %.3f s wrong %lu %s\n",
         rank, n, local, tId, tInit, tAr, bad, bad ? "FAIL" : "PASS");

  NCCL(ncclCommDestroy(comm));
  CUDA(cudaFree(send));
  CUDA(cudaFree(recv));
  return bad != 0;
}
