#include <ctf.hpp>
#include <chrono>
#include <float.h>

using namespace CTF;

void mttkrp(int n, World& dw) {
  int dimt[3] = {n, n, n};
  int dimt2[2] = {n, n};
  Tensor<double> B(3, false /* is_sparse */, dimt, dw);
  // Matrix<double> A(n, n, dw), C(n, n, dw), D(n, n, dw);
  Tensor<double> A(2, false, dimt2, dw), C(2, false, dimt2, dw), D(2, false, dimt2, dw);
  B.fill_random((double)0, (double)1);
  C.fill_random((double)0, (double)1);
  D.fill_random((double)0, (double)1);

  // Tensor<double>** matList = (Tensor<double>**)malloc(sizeof(Tensor<double>*) * 3);
  // matList[0] = &A;
  // matList[1] = &B;
  // matList[2] = &C;

  // for (int i = 0; i < B.order; i++) {
  //   std::cout << B.lens[i] << " " << matList[i]
  // }
  
  auto start = std::chrono::high_resolution_clock::now();
  A["il"] = B["ijk"] * C["jl"] * D["kl"];
  // MTTKRP<double>(&B, matList, 0, true);
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
