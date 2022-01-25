INCLUDES=-I/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-gcc-8.3.1/include/ LIB_PATH=-L/g/g15/yadav2/taco/legion/OpenBLAS/install/lib/ LIBS=-lblas CXX=mpicxx ./configure --install-dir=install
make -j
make -j install
