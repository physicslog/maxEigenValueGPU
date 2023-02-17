#### NVIDIA GPU Calculator to calculate largest eigenvalue of a huge matrix using shift-inverse method
Author: Damodar Rajbhandari (2023-Jan-01 - Last Update: 2023-Feb-17)

##### Usage
````
# Shifted-inverse power method using cuSolver
make
./maxeigenvalue mtxs/L11.mtx

# Power method using cuSparse and thrust
make mainpower
./maxeigenvaluepower mtxs/L11.mtx

# Compile Shifted-inverse power method, power method, and Spectra library
make all
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
./maxeigenvaluespectra mtxs/dL22.mtx
````

-----

#### Performance Comparison of GPU Power Method Vs Spectra Library
- Please install [hyperfine](https://github.com/sharkdp/hyperfine). It is a command-line benchmarking tool.
  - It can be installed via `conda` from the `conda-forge` channel:
    ````
    conda install -c conda-forge hyperfine
    ````

Here are the results:
- Using power method on GPU (unoptimized code, without using tolerance for convergence. See:  `computeMaxEigenvaluePowerMethod`)
  ````
  hyperfine './maxeigenvaluepower mtxs/dL22.mtx'
  ````
  - Results:
    ````
    Benchmark 1: ./maxeigenvaluepower mtxs/dL22.mtx
      Time (mean ± σ):     14.282 s ±  0.043 s    [User: 12.608 s, System: 1.569 s]
      Range (min … max):   14.241 s … 14.373 s    10 runs
    ````
- Using power method on GPU (optimized code, using tolerance for convergence. See `computeMaxEigenvaluePowerMethodOptimized`)
  ````
  hyperfine './maxeigenvaluepower mtxs/dL22.mtx'
  ````
  - Results:
    ````
    Benchmark 1: ./maxeigenvaluepower mtxs/dL22.mtx
      Time (mean ± σ):     13.782 s ±  0.038 s    [User: 12.112 s, System: 1.569 s]
      Range (min … max):   13.726 s … 13.873 s    10 runs
    ````
- Using Spectra library on CPU
  ````
  hyperfine './maxeigenvaluespectra mtxs/dL22.mtx'
  ````
  - Results:
    ````
    Benchmark 1: ./maxeigenvaluespectra mtxs/dL22.mtx
      Time (mean ± σ):      2.485 s ±  0.012 s    [User: 2.478 s, System: 0.007 s]
      Range (min … max):    2.466 s …  2.506 s    10 runs
    ````

----

#### Notes
- `maxeigenvalue ` which is based on `cusolverSpScsreigvs` doesnot work for larger matrices. For example: `mtx/dL22.mtx`.
- `maxeigenvaluepower` doesnot work for `mtx/dL00.mtx` if you use `computeMaxEigenvaluePowerMethod` but for others in `mtx/` directory; it works fine. This is because we set the initial vector `x_i` sets to 1.0 for all the elements. This is because our initial vector give rises to a orthogonal vector with eigenvector. Ideally choosing a random vector such that its norm is 1 and entries is mostly non-zero (because $Ax = 0$ if $x$ is $0$) allows the chance to decrease that our vector is orthogonal to the eigenvector. It is done in `computeMaxEigenvaluePowerMethodOptimized` function.
