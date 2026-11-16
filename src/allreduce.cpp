#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CUDA(x) do { cudaError_t e = x; if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define NCCL(x) do { ncclResult_t r = x; if (r != ncclSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, n, local;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  size_t count = argc > 1 ? strtoull(argv[1], nullptr, 0) : 1 << 24;

  MPI_Comm node;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
  MPI_Comm_rank(node, &local);
  CUDA(cudaSetDevice(local));  // before ncclCommInitRank!

  ncclUniqueId id;
  if (rank == 0) NCCL(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof id, MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm;
  NCCL(ncclCommInitRank(&comm, n, id, rank));

  size_t bytes = count * sizeof(float);
  std::vector<float> h(count, rank + 1.0f);
  float *send, *recv;
  CUDA(cudaMalloc(&send, bytes));
  CUDA(cudaMalloc(&recv, bytes));
  CUDA(cudaMemcpy(send, h.data(), bytes, cudaMemcpyHostToDevice));
  cudaStream_t s;
  CUDA(cudaStreamCreate(&s));
  NCCL(ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, s));
  CUDA(cudaStreamSynchronize(s));
  CUDA(cudaMemcpy(h.data(), recv, bytes, cudaMemcpyDeviceToHost));

  float want = n * (n + 1) / 2.0f;
  unsigned long bad = 0, badAll;
  for (float v : h) bad += v != want;
  MPI_Allreduce(&bad, &badAll, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    printf("allreduce nranks %d count %zu want %g wrong %lu %s\n", n, count, want, badAll, badAll ? "FAIL" : "PASS");

  NCCL(ncclCommDestroy(comm));
  CUDA(cudaFree(send));
  CUDA(cudaFree(recv));
  MPI_Finalize();
  return badAll != 0;
}
