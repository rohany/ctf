#include <ctf.hpp>
#include <chrono>
#include <float.h>
#include <functional>

using namespace CTF;

double benchmarkWithWarmup(int warmup, int numIter, std::function<void(void)> f) {
  for (int i = 0; i < warmup; i++) { f(); }
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < numIter; i++) { f(); }
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  return double(ms) / double(numIter);
}

std::vector<std::string> split(const std::string &str, const std::string &delim, bool keepDelim) {
  std::vector<std::string> results;
  size_t prev = 0;
  size_t next = 0;

  while ((next = str.find(delim, prev)) != std::string::npos) {
    if (next - prev != 0) {
      std::string substr = ((keepDelim) ? delim : "")
                         + str.substr(prev, next-prev);
      results.push_back(substr);
    }
    prev = next + delim.size();
  }

  if (prev < str.size()) {
    string substr = ((keepDelim) ? delim : "") + str.substr(prev);
    results.push_back(substr);
  }

  return results;
}

void spmv(int nIter, int warmup, std::string filename, std::vector<int> dims, World& dw) {
  Tensor<double> B(2, true /* is_sparse */, dims.data(), dw);
  Vector<double> a(dims[0], dw);
  Vector<double> c(dims[1], dw);
  c.fill_random(1.0, 1.0);

  B.read_sparse_from_file(filename.c_str());
  if (dw.rank == 0) {
    std::cout << "Read " << B.nnz_tot << " non-zero entries from the file." << std::endl;
  }

  auto avgMs = benchmarkWithWarmup(warmup, nIter, [&]() {
    a["i"] = B["ij"] * c["j"];
  });

  if (dw.rank == 0) {
    std::cout << "Average execution time: " << avgMs << " ms." << std::endl;
  }
}

void spmspv(int nIter, int warmup, std::string filename, std::string spmspvVecFile, std::vector<int> dims, World& dw) {
  Tensor<double> B(2, true /* is_sparse */, dims.data(), dw);
  Tensor<double> c(1, true /* is_sparse */, dims.data() + 1, dw);
  Vector<double> a(dims[0], dw);

  B.read_sparse_from_file(filename.c_str());
  if (dw.rank == 0) {
    std::cout << "Read " << B.nnz_tot << " non-zero entries from the file." << std::endl;
  }
  c.read_sparse_from_file(spmspvVecFile.c_str());
  if (dw.rank == 0) {
    std::cout << "Read " << c.nnz_tot << " non-zero entries from the file." << std::endl;
  }

  auto avgMs = benchmarkWithWarmup(warmup, nIter, [&]() {
    a["i"] = B["ij"] * c["j"];
  });

  if (dw.rank == 0) {
    std::cout << "Average execution time: " << avgMs << " ms." << std::endl;
  }
}

void spttv(int nIter, int warmup, std::string filename, std::vector<int> dims, World& dw) {
  Tensor<double> B(3, true /* is_sparse */, dims.data(), dw);
  Tensor<double> A(2, true /* is_sparse */, dims.data(), dw);
  Vector<double> c(dims[2], dw);
  c.fill_random(1.0, 1.0);

  B.read_sparse_from_file(filename.c_str());
  if (dw.rank == 0) {
    std::cout << "Read " << B.nnz_tot << " non-zero entries from the file." << std::endl;
  }

  auto avgMs = benchmarkWithWarmup(warmup, nIter, [&]() {
    A["ij"] = B["ijk"] * c["k"];
  });

  if (dw.rank == 0) {
    std::cout << "Average execution time: " << avgMs << " ms." << std::endl;
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

void spmm(int nIter, int warmup, std::string filename, std::vector<int> dims, World& dw, int jdim) {
  Tensor<double> B(2, true /* is_sparse */, dims.data(), dw);
  Matrix<double> A(dims[0], jdim, dw);
  Matrix<double> C(dims[1], jdim, dw);
  C.fill_random(1.0, 1.0);

  B.read_sparse_from_file(filename.c_str());
  if (dw.rank == 0) {
    std::cout << "Read " << B.nnz_tot << " non-zero entries from the file." << std::endl;
  }

  auto avgMs = benchmarkWithWarmup(warmup, nIter, [&]() {
    A["ij"] = B["ik"] * C["kj"];
  });

  if (dw.rank == 0) {
    std::cout << "Average execution time: " << avgMs << " ms." << std::endl;
  }
}

void mttkrp(int nIter, int warmup, std::string filename, std::vector<int> dims, World& dw, int ldim) {
  Tensor<double> B(3, true /* is_sparse */, dims.data(), dw);
  Matrix<double> A(dims[0], ldim, dw);
  Matrix<double> C(dims[1], ldim, dw);
  Matrix<double> D(dims[2], ldim, dw);
  C.fill_random(1.0, 1.0);
  D.fill_random(1.0, 1.0);

  B.read_sparse_from_file(filename.c_str());
  if (dw.rank == 0) {
    std::cout << "Read " << B.nnz_tot << " non-zero entries from the file." << std::endl;
  }
  
  Tensor<double>* mats[] = {&A, &C, &D};
  auto avgMs = benchmarkWithWarmup(warmup, nIter, [&]() {
    MTTKRP(&B, mats, 0, true);
  });

  if (dw.rank == 0) {
    std::cout << "Average execution time: " << avgMs << " ms." << std::endl;
  }
}

void sddmm(int nIter, int warmup, std::string filename, std::vector<int> dims, World& dw, int jdim) {
  Tensor<double> A(2, true /* is_sparse */, dims.data(), dw);
  Tensor<double> B(2, true /* is_sparse */, dims.data(), dw);
  Matrix<double> C(dims[0], jdim, dw);
  Matrix<double> D(jdim, dims[1], dw);
  C.fill_random(1.0, 1.0);
  D.fill_random(1.0, 1.0);
  
  B.read_sparse_from_file(filename.c_str());
  if (dw.rank == 0) {
    std::cout << "Read " << B.nnz_tot << " non-zero entries from the file." << std::endl;
  }

  auto avgMs = benchmarkWithWarmup(warmup, nIter, [&]() {
    // TODO (rohany): Figure out how to do the builtin function call here.
    // MTTKRP(&B, mats, 0, false);
  });

  if (dw.rank == 0) {
    std::cout << "Average execution time: " << avgMs << " ms." << std::endl;
  }
}

void innerprod(int nIter, int warmup, std::string filename, std::string tensorC, std::vector<int> dims, World& dw) {
  Tensor<double> B(3, true /* is_sparse */, dims.data(), dw);
  Tensor<double> C(3, true /* is_sparse */, dims.data(), dw);
  Scalar<double> a(dw);

  B.read_sparse_from_file(filename.c_str());
  if (dw.rank == 0) {
    std::cout << "Read " << B.nnz_tot << " non-zero entries from the file." << std::endl;
  }
  C.read_sparse_from_file(filename.c_str());
  if (dw.rank == 0) {
    std::cout << "Read " << C.nnz_tot << " non-zero entries from the file." << std::endl;
  }

  auto avgMs = benchmarkWithWarmup(warmup, nIter, [&]() {
    a[""] = B["ijk"] * C["ijk"];
  });

  if (dw.rank == 0) {
    std::cout << "Average execution time: " << avgMs << " ms." << std::endl;
  }
}

void spadd3(int nIter, int warmup, std::string filename, std::string tensorC, std::string tensorD, std::vector<int> dims, World& dw) {
  Tensor<double> A(2, true /* is_sparse */, dims.data(), dw);
  Tensor<double> B(2, true /* is_sparse */, dims.data(), dw);
  Tensor<double> C(2, true /* is_sparse */, dims.data(), dw);
  Tensor<double> D(2, true /* is_sparse */, dims.data(), dw);

  B.read_sparse_from_file(filename.c_str());
  if (dw.rank == 0) {
    std::cout << "Read " << B.nnz_tot << " non-zero entries from the file." << std::endl;
  }
  C.read_sparse_from_file(filename.c_str());
  if (dw.rank == 0) {
    std::cout << "Read " << C.nnz_tot << " non-zero entries from the file." << std::endl;
  }
  D.read_sparse_from_file(filename.c_str());
  if (dw.rank == 0) {
    std::cout << "Read " << D.nnz_tot << " non-zero entries from the file." << std::endl;
  }

  auto avgMs = benchmarkWithWarmup(warmup, nIter, [&]() {
    A["ij"] = B["ij"] + C["ij"] + D["ij"];
  });

  if (dw.rank == 0) {
    std::cout << "Average execution time: " << avgMs << " ms." << std::endl;
  }
}

int main(int argc, char** argv) {
  int nIter = 20, warmup = 10, mttkrpLDim = 32, spmmJDim = 32;
  std::string filename, bench = "spmv", tensorDims, spmspvVecFile, tensorC, tensorD;
  for (int i = 1; i < argc; i++) {
#define INT_ARG(argname, varname) do {      \
          if (!strcmp(argv[i], (argname))) {  \
            varname = atoi(argv[++i]);      \
            continue;                       \
          } } while(0);
#define STRING_ARG(argname, varname) do {      \
          if (!strcmp(argv[i], (argname))) {  \
            varname = std::string(argv[++i]);      \
            continue;                       \
          } } while(0);
    INT_ARG("-n", nIter);
    INT_ARG("-warmup", warmup);
    STRING_ARG("-tensor", filename);
    STRING_ARG("-bench", bench);
    STRING_ARG("-dims", tensorDims);
    STRING_ARG("-spmspvVec", spmspvVecFile);
    STRING_ARG("-tensorC", tensorC);
    STRING_ARG("-tensorD", tensorD);
    INT_ARG("-mttkrpLDim", mttkrpLDim);
    INT_ARG("-spmmJDim", spmmJDim);
#undef INT_ARG
#undef STRING_ARG
  }

  if (filename.empty()) {
    std::cout << "Please provide an input filename." << std::endl;
    return -1;
  }

  if (tensorDims.empty()) {
    std::cout << "Must provide tensor dims." << std::endl;
    return -1;
  }

  auto dimsStr = split(tensorDims, ",", false /* keepDelim */);
  std::vector<int> dims;
  for (auto it : dimsStr) {
    dims.push_back(atoi(it.c_str()));
  }

  MPI_Init(&argc, &argv);
  int np;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  World dw;
  int retVal = 0;
  if (bench == "spmv") {
    spmv(nIter, warmup, filename, dims, dw);
  } else if (bench == "spmspv") {
    if (spmspvVecFile.empty()) {
      std::cout << "Must provide sparse vector." << std::endl;
      retVal = -1;
    } else {
      spmspv(nIter, warmup, filename, spmspvVecFile, dims, dw);
    }
  } else if (bench == "spmm") {
    spmm(nIter, warmup, filename, dims, dw, spmmJDim);
  } else if (bench == "spttv" ) {
    spttv(nIter, warmup, filename, dims, dw);
  } else if (bench == "mttkrp") {
    mttkrp(nIter, warmup, filename, dims, dw, mttkrpLDim);
  } else if (bench == "innerprod") {
    if (tensorC.empty()) {
      std::cout << "Must provide tensorC." << std::endl;
      retVal = -1;
    } else {
      innerprod(nIter, warmup, filename, tensorC, dims, dw);
    }
  } else if (bench == "spadd3") {
    if (tensorC.empty() || tensorD.empty()) {
      std::cout << "Must provide tensorC and tensorD." << std::endl;
      retVal = -1;
    } else {
      spadd3(nIter, warmup, filename, tensorC, tensorD, dims, dw);
    }
  } else {
    std::cout << "Unknown benchmark name: " << bench << std::endl;
    retVal = -1;
  }
  MPI_Finalize();
  return retVal;
}

