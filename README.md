#### NVIDIA GPU Calculator to calculate largest eigenvalue of a huge matrix using shift-inverse method
Author: Damodar Rajbhandari (2022-Jan-01)

##### Usage
````
make
./maxeigenvalue mtxs/L11.mtx
````

##### Software Dependencies
- [nvcc compiler supporting c++14](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) to compile the codebase
- [cuSparse (in CUDA Toolkit)](https://docs.nvidia.com/cuda/cusparse/index.html) for sparse matrix operations
- [cuSolver (in CUDA Toolkit)](https://docs.nvidia.com/cuda/cusolver/index.html) to find largest eigenvalue
- [NVIDIA Nsight for vscode extension (Optional)](https://developer.nvidia.com/nsight-visual-studio-code-edition) for debugging and profiling purposes

##### Hardware Requirements
- Code ran on NVIDIA GeForce RTX 2080 Ti and CUDA Version 11.7. Check yours using `nvidia-smi -q` command.