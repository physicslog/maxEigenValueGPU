# Damodar Rajbhandari (2023)

CUDA_TOOLKIT     := $(shell dirname $$(command -v nvcc))/..
INC              := -I$(CUDA_TOOLKIT)/include
LIBS         		 := -lcusparse -lcusolver
EIGS_INC         := -I /usr/include/eigen3

# `-g -G` option pair must be passed to nvcc to enable debugging using CUDA-GDB
main: maxeigenvalue.cu
	nvcc -std=c++14 -g -G $(INC) maxeigenvalue.cu -o maxeigenvalue $(LIBS)

mainspectra: maxeigenvaluespectra.cpp
	g++ -std=c++14 ${EIGS_INC} maxeigenvaluespectra.cpp -o maxeigenvaluespectra

clean:
	rm -f maxeigenvalue maxeigenvaluespectra

.PHONY: main mainspectra clean