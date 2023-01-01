#### NVIDIA GPU Calculator to calculate largest eigenvalue of a huge matrix using shift-inverse method

##### Usage
````
make
./maxeigenvalue L11.mtx
````

##### Software Dependencies
- [nvcc compiler supporting c++14](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) to compile the codebase
- [cuSparse (in CUDA Toolkit)](https://docs.nvidia.com/cuda/cusparse/index.html) for sparse matrix operations
- [cuSolver (in CUDA Toolkit)](https://docs.nvidia.com/cuda/cusolver/index.html) for finding maximum eigenvalue

##### Hardware Requirements
- Simulation ran on NVIDIA GeForce RTX 2080 Ti and CUDA Version 11.7. Check yours using `nvidia-smi -q` command.