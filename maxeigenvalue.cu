// Damodar Rajbhandari (2023-Jan-01)
// Code compilation: make
// Usage: ./maxeigenvalue <path-to-matrix-market-file>

// C++ DEPENDENCIES
#include <iostream>
#include <fstream>
#include <sstream>

// CUDA TOOLKIT DEPENDENCIES
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/random.h>

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


// Generate random number in the range [0, 1)
struct genRandomNumber {
    __device__
    float operator () (int idx) {
        thrust::default_random_engine randGen;
        thrust::uniform_real_distribution<float> uniDist;
        randGen.discard(idx);
        return uniDist(randGen);
  }
};

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
    std::cout << "Error opening file: " << filepath << std::endl;
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
    std::cout << "Only real and integer valued matrices are supported" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Get symmetry (matrix needs to be square matrix)
  bool is_symmetric = false;
  if (substr[4] == "symmetric") {
    is_symmetric = true;
  } else if (substr[4] == "general") {
    is_symmetric = false;
  } else {
    std::cout << "Only symmetric and general matrices are supported" << std::endl;
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

float computeMaxEigenvalue(const CSThrust& M) {
  assert(M.format == "CSR");  // We only use CSR format
  assert(M.m == M.n);
  cusolverSpHandle_t solver_handle = NULL;
  CHECK_CUSOLVER( cusolverSpCreate(&solver_handle) )
  cusparseMatDescr_t M_descr;
  CHECK_CUSPARSE( cusparseCreateMatDescr(&M_descr) )

  //! @note CuSolverSp only supports the matrix is general type. i.e. CUSPARSE_MATRIX_TYPE_GENERAL
  //!       So, if the Matrix is stored only in the upper or lower triangular part, then we need to
  //!       extend into its missing lower or upper part, otherwise the result would be wrong.
  //!       Fortunately, we don't need to do this, because cusparseSpGEMM() will automatically
  //!       store its result in the general format.
  //!       See: https://docs.nvidia.com/cuda/cusolver/index.html#cusolversp-t-csreigvsi
  CHECK_CUSPARSE( cusparseSetMatType(M_descr, CUSPARSE_MATRIX_TYPE_GENERAL) )
  CHECK_CUSPARSE( cusparseSetMatIndexBase(M_descr, CUSPARSE_INDEX_BASE_ZERO) )

  // Define required intial values and threshold
  float tol = 1e-3;  // tolerance to determine the convergence
  int max_iter = 1000;  // maximum number of iterations
  //! @note To set the simple version of upper bound of the largest eigenvalue of the (graph-)Laplacian.
  //!       We use Gershgorin circle theorem such that
  //!       |lambda_max_guess| = d_i^max + R_i where,
  //!       d_i^max is largest degree of the vertex at ith vertex, and
  //!       R_i is the radius of the Gershgorin circle at ith vertex := \sum_{j \ne i} |M_ij|
  //!       For simplicity, we replace R_i by 2*(nnz - m)/m. lambda_max_guess doesn't needs to be perfect because it's
  //!       just a initial guess, but it should be as close as possible to the actual max. eigenvalue.
  //!       Let me show you how I got this formula:
  //!       1. Probability of nnz elements on the (i, j) excluding diagonal (i.e. i != j) is (nnz - m)/(m^2 - m) = (nnz - m)/(m(m - 1)).
  //!       2. Total probable nnz elements on the ith row excluding diagonal ii is (m - 1) * (nnz - m)/(m(m -1)) = (nnz - m)/m.
  //!       3. Total probable  nnz elements on the ith column excluding diagonal ii is (m - 1) * (nnz - m)/(m(m -1)) = (nnz - m)/m.
  //!       4. Total probable nnz elements either on the row or column excluding its diagonal element is 2*(nnz - m)/m.
  float lambda_max_guess = *thrust::max_element(M.values.begin(), M.values.end()) + ((M.nnz - M.m)/M.m)*2;

  std::cout << "Initial guess for the largest eigenvalue: " << lambda_max_guess << std::endl;

  thrust::device_vector<float> x0(M.m);  // Get the random initial vector
  thrust::transform(thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(M.m),
                    x0.begin(),
                    genRandomNumber());

  thrust::device_vector<float> x(M.m, 0);  // the eigenvector

  // Setting the variable for the largest eigenvalue to store
  //! @note cuSolver has incorrect documentation for cusolverSpScsreigvsi. mu (i.e. lambda_max_computed)
  //!       should be a device pointer, not a host pointer.
  float *d_lambda_max_computed;
  CHECK_CUDA(cudaMalloc(&d_lambda_max_computed, sizeof(float)));
  CHECK_CUDA(cudaMemset(d_lambda_max_computed, 0, sizeof(float)));

  // Get eigenvalue and eigenvector near to the upper bound of the largest eigenvalue

  CHECK_CUSOLVER( cusolverSpScsreigvsi(solver_handle,
                                        M.m, M.nnz,
                                        M_descr,
                                        thrust::raw_pointer_cast(M.values.data()),
                                        thrust::raw_pointer_cast(M.pointers.data()),
                                        thrust::raw_pointer_cast(M.indices.data()),
                                        lambda_max_guess,
                                        thrust::raw_pointer_cast(x0.data()),
                                        max_iter, tol,
                                        d_lambda_max_computed,
                                        thrust::raw_pointer_cast(x.data())) )  // This line throws segmentation fault for larger matrix sizes. RESCUE ME?

  // Copy to host
  float lambda_max_computed;
  CHECK_CUDA(cudaMemcpy(&lambda_max_computed, d_lambda_max_computed, sizeof(float), cudaMemcpyDeviceToHost));

  // Destroy the descriptor, handle
  CHECK_CUSPARSE( cusparseDestroyMatDescr(M_descr) )
  // CHECK_CUSPARSE( cusparseDestroy(sparse_handle) )
  CHECK_CUSOLVER( cusolverSpDestroy(solver_handle) )

  return lambda_max_computed;
}

int main(int argc, char** argv) {
  std::string mtx_filepath;
  if (argc > 1) {
    mtx_filepath = argv[1];
  } else {
    std::cout << "Please provide a path to a matrix market file." << std::endl;
    return EXIT_FAILURE;
  }

  CSThrust M;  // Create a sparse matrix instance
  readMTXFile2CSR(mtx_filepath, M);  // Read the matrix market file and convert it to csr
  float lambda_max = computeMaxEigenvalue(M);  // Compute the largest eigenvalue
  std::cout << "Max eigenvalue: " << lambda_max << std::endl;

  return EXIT_SUCCESS;
}