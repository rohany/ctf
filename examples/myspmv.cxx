#include <ctf.hpp>
#include <chrono>
#include <float.h>

using namespace CTF;

void spmv(int nIter, int warmup, std::string filename, World& dw) {
  // TODO (rohany): I don't know what's the best way to get the 
  //  dimensions of the tensor (which is encoded in the file already...).
  // int x = 1102824;
  // int y = x;
  // int x = 22744080;
  // int y = 22744080;
  // int x = 2902330;
  // int y = 2143368; 
  // int z = 25495389;
  int x = 12092;
  int y = 9184;
  int z = 28818;
  int lens[] = {x, y, z};
  Tensor<double> B(3, true /* is_sparse */, lens, dw);
  // Tensor<double> B(2, true /* is_sparse */, lens, dw);
  Vector<double> a(x, dw);
  Vector<double> c(z, dw);
  // Needed for the MTTKRP Cast.
  Vector<double> d(z, dw);
  // Vector<double> c(y, dw);
  //
  c.fill_random(1.0, 1.0);
  d.fill_random(1.0, 1.0);
  a.fill_random(0.0, 0.0);

  auto compute = [&]() {
    Tensor<double>* mats[] = {&a, &c, &d};
    MTTKRP(&B, mats, 0, false /* unsure... */);
    // a["i"] = B["ijk"] * c["k"];
    // a["i"] = B["ij"] * c["j"];
  };

  B.read_sparse_from_file(filename.c_str());
  if (dw.rank == 0) {
    std::cout << "Read " << B.nnz_tot << " non-zero entries from the file." << std::endl;
  }

  for (int i = 0; i < warmup; i++) { compute(); }
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < nIter; i++) { compute(); }
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  if (dw.rank == 0) {
    std::cout << "Average execution time: " << (double(ms) / double(nIter)) << " ms." << std::endl;
  }

  // TODO (rohany): For some reason this call to norm1 is hanging. Let's compute the norm by hand.
  // if (dw.rank == 0) {
  //   std::cout << a.norm1() << std::endl;
  // }
  // double* localData = NULL;
  // int64_t numValues = 0;
  // a.get_all_data(&numValues, &localData);
  // if (dw.rank == 0) {
  //   std::cout << numValues << " " << x << std::endl;
  // }
  // double norm1 = 0.0;
  // for (int i = 0; i < numValues; i++) { 
  //   norm1 += localData[i];
  //   if (dw.rank == 0) {
  //     std::cout << localData[i] << std::endl;
  //   }
  // }
  // if (dw.rank == 0) {
  //   std::cout << "norm1: " << norm1 << std::endl;
  // }
}

int main(int argc, char** argv) {
  int nIter = -1, warmup = -1;
  std::string filename;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0) {
      nIter = atoi(argv[++i]);
      continue;
    }
    if (strcmp(argv[i], "-warmup") == 0) {
      warmup = atoi(argv[++i]);
      continue;
    }
    if (strcmp(argv[i], "-tensor") == 0) {
      filename = std::string(argv[++i]);
      continue;
    }
  }

  if (nIter == -1 || warmup == -1 || filename.empty()) {
    std::cout << "provide all inputs." << std::endl;
    return -1;
  }

  MPI_Init(&argc, &argv);
  int np;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  {
    World dw;
    spmv(nIter, warmup, filename, dw);
  }
  MPI_Finalize();
  return 0;
}

