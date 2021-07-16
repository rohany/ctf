#include <ctf.hpp>
#include <chrono>
#include <float.h>
using namespace CTF;

void ttmc(int n, int np, int procsPerNode, World& dw) {
  int dimt[3] = {n, n, n};
  Tensor<double> A(3, false /* is_sparse */, dimt, dw), B(3, false /* is_sparse */, dimt, dw);
  Matrix<double> C(n, n, dw);
  B.fill_random((double)0, (double)1);
  C.fill_random((double)0, (double)1);
 
  std::vector<size_t> times;
  for (int i = 0; i < 10; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    A["ijl"] = B["ijk"] * C["kl"];
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (dw.rank == 0) {
      std::cout << "Execution time: " << ms << " ms." << std::endl;
    }
    times.push_back(ms);
  } 

  size_t flops = size_t(n) * (size_t(n) * size_t(n) * (2 * size_t(n) - 1));

  size_t sum = 0;
  for (auto t : times) {
    sum += t;
  }
  double avg = double(sum) / double(times.size());

  auto secs = avg / 1e3;
  auto gflop = double(flops) / 1e9;
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
    ttmc(n, np, procsPerNode, dw);
  }
  MPI_Finalize();
  return 0;
}
