// Damodar Rajbhandari (2023-Feb-16)
// Code compilation: make mainpower
// Usage: ./maxeigenvaluepower <path-to-matrix-market-file>

// C++ DEPENDENCIES
#include <iostream>
#include <fstream>
#include <sstream>

// CUDA TOOLKIT DEPENDENCIES
#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>

// Terminal output color (just for cosmetic purpose)
#define RST  "\x1B[37m"  // Reset color to white
#define KGRN  "\033[0;32m"   // Define green color
#define RD "\x1B[31m"  // Define red color
#define FGRN(x) KGRN x RST  // Define compiler function for green color
#define FRD(x) RD x RST  // Define compiler function for red color

// To check if CUDA API calls are successful
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

// To check if cuSPARSE API calls are successful
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

// To check if cuSOLVER API calls are successful
#define CHECK_CUSOLVER(func)                                                   \
{                                                                              \
    cusolverStatus_t status = (func);                                          \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
        printf("CUSOLVER API failed at line %d with error: %s (%d)\n",         \
               __LINE__, status, status);                                      \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}


struct CSThrust {
  int m;  // number of rows
  int n;  // number of columns
  int nnz;  // number of non-zero elements
  std::string format; // Either "CSC" or "CSR"
  thrust::device_vector<int> pointers;  // can be column pointer (also called column offsets) or row pointer (also called row offsets)
  thrust::device_vector<int> indices;  // can be row indices or column indices
  thrust::device_vector<float> values;  // can be cscValues or csrValues
};

// Read a matrix market file and return a CSR matrix
// See https://math.nist.gov/MatrixMarket/formats.html for more details
void readMTXFile2CSR(const std::string& filepath, CSThrust& csr) {
  std::ifstream mtxfile(filepath.c_str(), std::ios::in);
  if (!mtxfile.is_open()) {
    std::cout << FRD("[ERROR]: ") << "Error opening file: " << filepath << std::endl;
    exit(EXIT_FAILURE);
  }

  // Reading header
  std::string header;
  std::getline(mtxfile, header);
  assert(mtxfile.good());

  std::stringstream formatheader(header);
  std::string substr[5];
  formatheader >> substr[0] >> substr[1] >> substr[2] >> substr[3] >> substr[4];
  assert(substr[0] == "%%MatrixMarket");
  assert(substr[1] == "matrix");
  assert(substr[2] == "coordinate");

  if (substr[3].compare("complex") == 0) {
    std::cout << FRD("[ERROR]: ") << "Only real and integer valued matrices are supported" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Get symmetry (matrix needs to be square matrix)
  bool is_symmetric = false;
  if (substr[4] == "symmetric") {
    is_symmetric = true;
  } else if (substr[4] == "general") {
    is_symmetric = false;
  } else {
    std::cout << FRD("[ERROR]: ") <<  "Only symmetric and general matrices are supported" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Ignore comments afterwards
  while (mtxfile.peek() == '%') {
    mtxfile.ignore(2048, '\n');
  }

  // Read nrows, ncols, nnz
  int nrows, ncols, nnz;
  mtxfile >> nrows >> ncols >> nnz;

  // Set and resize CSR variables
  csr.format = "CSR";
  csr.m = nrows;
  csr.n = ncols;

  // Symmetric matrix is a square matrix
  if (is_symmetric) {
    assert(nrows == ncols);
    csr.nnz = 2* nnz - nrows;
  } else {
    csr.nnz = nnz;
  }

  // Read the matrix from mtx file, if it's symmetric, then recreate upper triangular part.
  // Store in row-major order
  std::vector<float> values(nrows * ncols, 0.0);
  int row, col;
  float val;
  int diag_count = 0;
  for (int coeff = 0; coeff < nnz; coeff++) {
    mtxfile >> row >> col >> val;
    if (row == col) {
      diag_count++;
      values[(row - 1) * ncols + (col - 1)] = val;
      if (is_symmetric) {
        values[(col - 1) * ncols + (row - 1)] = val;
      }
    } else {
      values[(row - 1) * ncols + (col - 1)] = val;
      if (is_symmetric) {
        values[(col - 1) * ncols + (row - 1)] = val;
      }
    }
  }

  assert(diag_count == nrows);
  mtxfile.close();

  // Convert to CSR
  csr.pointers.resize(csr.m + 1);
  csr.indices.resize(csr.nnz);
  csr.values.resize(csr.nnz);

  // Setting row offset start with 0 index
  csr.pointers[0] = 0;

  // Extract out the CSR variables
  int nnz_idx = 0;
  for (int row = 0; row < csr.m; row++) {
    for (int col = 0; col < csr.n; col++) {
      if (values[row * csr.n + col] != 0.0) {
        csr.indices[nnz_idx] = col;
        csr.values[nnz_idx] = values[row * csr.n + col];
        nnz_idx++;
      }
    }
    csr.pointers[row + 1] = nnz_idx;
  }
}


float computeMaxEigenvaluePowerMethod(CSThrust& M, int max_iter) {
  assert(M.format == "CSR");  // We only use CSR format
  assert(M.m == M.n);

  // Initialize x_i to [1 1 ... 1]^T
  thrust::device_vector<float> x_i(M.m, 1.0f), x_k(M.m, 0.0f);
  float max_eigenvalue;

  // CUSPARSE APIs
  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matM;
  cusparseDnVecDescr_t xi, xk;
  void *dBuffer = NULL;
  size_t bufferSize = 0;
  float alpha = 1.0f;
  float beta = 0.0f;

  CHECK_CUSPARSE( cusparseCreate(&handle) )

  CHECK_CUSPARSE( cusparseCreateCsr(&matM, M.m, M.n, M.nnz,
                                   thrust::raw_pointer_cast(M.pointers.data()),
                                   thrust::raw_pointer_cast(M.indices.data()),
                                   thrust::raw_pointer_cast(M.values.data()),
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

  CHECK_CUSPARSE( cusparseCreateDnVec(&xi, M.m, thrust::raw_pointer_cast(x_i.data()), CUDA_R_32F) )
  CHECK_CUSPARSE( cusparseCreateDnVec(&xk, M.m, thrust::raw_pointer_cast(x_k.data()), CUDA_R_32F) )

  CHECK_CUSPARSE( cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha, matM, xi, &beta, xk, CUDA_R_32F,
                                          CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )

  CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

  // Power iteration method
  for (int i = 0; i < max_iter; i++) {
    // Compute x_k = A * x_i; generates Krylov subspace
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matM, xi, &beta, xk, CUDA_R_32F,
                                 CUSPARSE_MV_ALG_DEFAULT, dBuffer) )

    // Compute the L2 norm of x_k
    float norm = std::sqrt(thrust::inner_product(x_k.begin(), x_k.end(), x_k.begin(), 0.0f));

    // Normalize x_k and update x_i
    // thrust::transform(x_k.begin(), x_k.end(), x_i.begin(), x_i.begin(), thrust::placeholders::_1 / norm);  // redundant
    thrust::transform(x_k.begin(), x_k.end(), x_i.begin(), thrust::placeholders::_1 / norm);
  }

  // Compute the maximum eigenvalue
  CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matM, xi, &beta, xk, CUDA_R_32F,
                               CUSPARSE_MV_ALG_DEFAULT, dBuffer) )

  max_eigenvalue = thrust::inner_product(x_i.begin(), x_i.end(), x_k.begin(), 0.0f);

  // Destroy the handle and descriptors
  CHECK_CUSPARSE( cusparseDestroySpMat(matM) )
  CHECK_CUSPARSE( cusparseDestroyDnVec(xi) )
  CHECK_CUSPARSE( cusparseDestroyDnVec(xk) )
  CHECK_CUSPARSE( cusparseDestroy(handle) )
  CHECK_CUDA( cudaFree(dBuffer) )

  return max_eigenvalue;
}


float computeMaxEigenvaluePowerMethodOptimized(CSThrust& M, int max_iter) {
  assert(M.format == "CSR");  // We only use CSR format
  assert(M.m == M.n);

  // Initialize x_i to [1 1 ... 1]^T
  thrust::device_vector<float> x_i(M.m, 1.0f), x_k(M.m, 0.0f);

  // CUSPARSE APIs
  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matM;
  cusparseDnVecDescr_t xi, xk;
  void *dBuffer = NULL;
  size_t bufferSize = 0;
  float alpha = 1.0f;
  float beta = 0.0f;

  CHECK_CUSPARSE( cusparseCreate(&handle) )

  CHECK_CUSPARSE( cusparseCreateCsr(&matM, M.m, M.n, M.nnz,
                                   thrust::raw_pointer_cast(M.pointers.data()),
                                   thrust::raw_pointer_cast(M.indices.data()),
                                   thrust::raw_pointer_cast(M.values.data()),
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

  CHECK_CUSPARSE( cusparseCreateDnVec(&xi, M.m, thrust::raw_pointer_cast(x_i.data()), CUDA_R_32F) )
  CHECK_CUSPARSE( cusparseCreateDnVec(&xk, M.m, thrust::raw_pointer_cast(x_k.data()), CUDA_R_32F) )

  CHECK_CUSPARSE( cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha, matM, xi, &beta, xk, CUDA_R_32F,
                                          CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )

  CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

  float max_eigenvalue(0.0f), max_eigenvalue_prev(0.0f);
  float tol = 1e-6;  // tolerance for convergence
  int itr = 0;
  // Power iteration method
  while (itr < max_iter) {
    // Compute x_k = A * x_i; generates Krylov subspace
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matM, xi, &beta, xk, CUDA_R_32F,
                                 CUSPARSE_MV_ALG_DEFAULT, dBuffer) )

    // Compute the L2 norm of x_k
    float norm = std::sqrt(thrust::inner_product(x_k.begin(), x_k.end(), x_k.begin(), 0.0f));

    // Normalize x_k and update x_i
    thrust::transform(x_k.begin(), x_k.end(), x_i.begin(), thrust::placeholders::_1 / norm);

    // Compute the maximum eigenvalue
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matM, xi, &beta, xk, CUDA_R_32F,
                                CUSPARSE_MV_ALG_DEFAULT, dBuffer) )

    max_eigenvalue = thrust::inner_product(x_i.begin(), x_i.end(), x_k.begin(), 0.0f);

    if (std::abs(max_eigenvalue - max_eigenvalue_prev) < tol) {
      std::cout << FGRN("[SUCCESS]: ") << "Converged at iterations: " << itr << std::endl;
      return max_eigenvalue;
    }

    max_eigenvalue_prev = max_eigenvalue;
    itr++;
  }

  // Destroy the handle and descriptors
  CHECK_CUSPARSE( cusparseDestroySpMat(matM) )
  CHECK_CUSPARSE( cusparseDestroyDnVec(xi) )
  CHECK_CUSPARSE( cusparseDestroyDnVec(xk) )
  CHECK_CUSPARSE( cusparseDestroy(handle) )
  CHECK_CUDA( cudaFree(dBuffer) )

  std::cout << FRD("[NOTE]: ") << "Maximum number of iterations reached." << std::endl;  // no convergence
  return max_eigenvalue;
}

int main(int argc, char** argv) {
  std::string mtx_filepath;
  if (argc > 1) {
    mtx_filepath = argv[1];
  } else {
    std::cout << FRD("[ERROR]: ") << "Please provide a path to a matrix market file." << std::endl;
    return EXIT_FAILURE;
  }

  CSThrust M;  // Create a sparse matrix instance
  readMTXFile2CSR(mtx_filepath, M);  // Read the matrix market file and convert it to csr
  // float lambda_max = computeMaxEigenvaluePowerMethod(M, 1000);  // Compute the largest eigenvalue
  float lambda_max = computeMaxEigenvaluePowerMethodOptimized(M, 1000);  // Compute the largest eigenvalue
  std::cout << FGRN("[SUCCESS]: ") << "Max eigenvalue: " << lambda_max << std::endl;

  return EXIT_SUCCESS;
}