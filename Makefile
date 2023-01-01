# Damodar Rajbhandari (2023)

CUDA_TOOLKIT     := $(shell dirname $$(command -v nvcc))/..
INC              := -I$(CUDA_TOOLKIT)/include
LIBS         		 := -lcusparse -lcusolver

# `-g -G` option pair must be passed to nvcc to enable debugging using CUDA-GDB
main: maxeigenvalue.cu
			nvcc -std=c++14 -g -G $(INC) maxeigenvalue.cu -o maxeigenvalue $(LIBS)

.PHONY: main