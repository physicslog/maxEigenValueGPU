# Damodar Rajbhandari (2023)

CUDA_TOOLKIT     := $(shell dirname $$(command -v nvcc))/..
INC              := -I$(CUDA_TOOLKIT)/include
LIBS         		 := -lcusparse -lcusolver

main: maxeigenvalue.cu
			nvcc -std=c++14 $(INC) maxeigenvalue.cu -o maxeigenvalue $(LIBS)

.PHONY: main