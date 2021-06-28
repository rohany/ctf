#include <ctf.hpp>
#include <chrono>
#include <float.h>
using namespace CTF;

void mttkrp(int n, World& dw) {
  int dimt[3] = {n, n, n};
  Tensor<double> B(3, false /* is_sparse */, dimt, dw);
  Matrix<double> A(n, n, dw), C(n, n, dw), D(n, n, dw);
  B.fill_random((double)0, (double)1);
  C.fill_random((double)0, (double)1);
  D.fill_random((double)0, (double)1);
  
  auto start = std::chrono::high_resolution_clock::now();
  A["il"] = B["ijk"] * C["jl"] * D["kl"];
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  if (dw.rank == 0) {
    std::cout << "Execution time: " << ms << " ms." << std::endl;
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  {
    World dw;
    for (int i = 0; i < 10; i++) {
    	mttkrp(512, dw);
    }
  }
  MPI_Finalize();
  return 0;
}
