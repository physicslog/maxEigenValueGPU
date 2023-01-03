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

----

##### CPU Result Comparison
- Used [Spectra version 1.0.1](https://spectralib.org) on the top of [Eigen3 version 3.4.0](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- Code ran on Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz with 125GB RAM and Arch GNU/Linux x86-64 with Linux kernel: 5.15.41-1-lts. Check yours using these commands: `lscpu` to get CPU details, `free -g -h -t` to get RAM details, and `cat /etc/os-release` OS details.

###### Usage
````
make mainspectra
./maxeigenvalue mtxs/dL22.mtx
````

-----