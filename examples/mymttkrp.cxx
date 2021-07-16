#include <ctf.hpp>
#include <chrono>
#include <float.h>

using namespace CTF;

size_t gemmFlops(size_t M, size_t N, size_t K) {
  return M * N * (2 * K - 1);
}

size_t mttkrpFlops(size_t I, size_t J, size_t K, size_t L) {
  return I * gemmFlops(J, K, L) + 2 * (I * J * L);
}

void mttkrp(size_t n, int np, int procsPerNode, World& dw) {
  int dimt[3] = {int(n), int(n), int(n)};
  int dimt2[2] = {int(n), int(n)};
  Tensor<double> B(3, false /* is_sparse */, dimt, dw);
  Tensor<double> A(2, false, dimt2, dw), C(2, false, dimt2, dw), D(2, false, dimt2, dw);
  B.fill_random((double)0, (double)1);
  C.fill_random((double)0, (double)1);
  D.fill_random((double)0, (double)1);
  
  std::vector<size_t> times;
  for (int i = 0; i < 10; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    A["il"] = B["ijk"] * C["jl"] * D["kl"];
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (dw.rank == 0) {
      std::cout << "Execution time: " << ms << " ms." << std::endl;
    }
    times.push_back(ms);
  }

  size_t sum = 0;
  for (auto t : times) {
    sum += t;
  }
  double avg = double(sum) / double(times.size());
  auto secs = avg / 1e3;
  auto flops = mttkrpFlops(n, n, n, n);
  auto gflop = flops / 1e9;
  auto gflops = gflop / secs;
  auto nodes = np / procsPerNode;
  auto gflopsPerNode = gflops / nodes;
  if (dw.rank == 0) {
    printf("On %ld nodes achieved GFLOPS per node: %lf.\n", nodes, gflopsPerNode);
  }
}

int main(int argc, char** argv) {
  int n = -1;
  int procsPerNode = 1;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0) {
      n = atoi(argv[++i]);
      continue;
    }
    if (strcmp(argv[i], "-procsPerNode") == 0) {
      procsPerNode = atoi(argv[++i]);
      continue;
    }
  }

  if (n == -1) {
    std::cout << "Please provide an input size with -n." << std::endl;
    return 1;
  }

  MPI_Init(&argc, &argv);
  int np;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  {
    World dw;
    mttkrp(n, np, procsPerNode, dw);
  }
  MPI_Finalize();
  return 0;
}
