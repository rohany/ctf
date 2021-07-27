#include <ctf.hpp>
#include <chrono>
#include <float.h>
using namespace CTF;

void ttv(size_t n, int np, int procsPerNode, World& dw) {
  int dimt[3] = {int(n), int(n), int(n)};
  Tensor<double> B(3, false /* is_sparse */, dimt, dw);
  Tensor<double> C(3, false /* is_sparse */, dimt, dw);
  Scalar<double> A(dw);
  B.fill_random((double)0, (double)1);
  C.fill_random((double)0, (double)1);
  
  std::vector<size_t> times;
  for (int i = 0; i < 10; i++) {
    A.set_val(double(0));
    auto start = std::chrono::high_resolution_clock::now();
    A[""] += B["ijk"] * C["ijk"];
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
  size_t elems = 2 *  n * n * n;
  size_t bytes = elems * sizeof(double);
  double gbytes = double(bytes) / 1e9;
  double bw = gbytes / secs;
  auto nodes = np / procsPerNode;
  if (dw.rank == 0) {
    printf("On %ld nodes achieved GB/s BW per node: %lf.\n", nodes, bw / double(nodes));
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
    ttv(n, np, procsPerNode, dw);
  }
  MPI_Finalize();
  return 0;
}
