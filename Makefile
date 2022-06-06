CC:=mpic++

TF_DIR:=(the directory of your TensorFlow dynamic library)
MKL_DIR:=(the directory of your Intel mkl library)
INC_DIR:=  -I$(TF_DIR)/include 
LINK_DIR:= -L$(TF_DIR)/lib -L$(MKL_DIR)/lib/intel64

LINK_LIB:=-ltensorflow_cc -ltensorflow_framework -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64 -lmkl_rt -lmkl_core -lmkl_gnu_thread -lgomp -Wl,-rpath=$(TF_DIR)/lib

CFLAGS:=-std=c++11 -g -O3 -Wall -fPIC -fopenmp

train-test: train-test.o scalapack.o
	$(CC) $^ -o $@ $(LINK_DIR) $(LINK_LIB)
train-test.o: train-test.cc
	$(CC) -c -o $@ $< $(CFLAGS) $(INC_DIR)
scalapack.o: scalapack.cc
	$(CC) -c -o $@ $< $(CFLAGS)
all: train-test
clean:
	rm train-test *.o
