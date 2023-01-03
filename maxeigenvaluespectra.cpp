// Damodar Rajbhandari (2023-Jan-03)
// Code compilation: make mainspectra
// Usage: ./maxeigenvaluespectra <path-to-matrix-market-file>

// C++ DEPENDENCIES
#include <iostream>

// EXTERNAL DEPENDENCIES: Eigen3, Spectra
#include <Eigen/SparseCore>  // used to create sparse matrix
#include <unsupported/Eigen/SparseExtra>  // used to load mtx matrix
#include <Spectra/SymEigsSolver.h>  // used to calculate max eigenvalue
#include <Spectra/MatOp/SparseSymMatProd.h>  // used to create matrix operation object


// Terminal output color (just for cosmetic purpose)
#define RST  "\x1B[37m"  // Reset color to white
#define KGRN  "\033[0;32m"   // Define green color
#define RD "\x1B[31m"  // Define red color
#define FGRN(x) KGRN x RST  // Define compiler function for green color
#define FRD(x) RD x RST  // Define compiler function for red color

/**
 * @brief Calculates the maximum eigenvalue of a sparse matrix using Spectra lib.
 * @param L The sparse square matrix
 * @param ncv
 * @return max_eigen_value
 * @note Spectra (https://spectralib.org/doc/symeigsbase_8h_source) uses
 * 	     Lanczos method (https://en.wikipedia.org/wiki/Lanczos_algorithm) to calculate
 *       eigenvalue. The number of Lanczos vectors (ncv) is a parameter that can be
 *       tuned to improve the performance of the algorithm. The default value is 20.
 */
float calcMaxEigenvalue(const Eigen::SparseMatrix<float> &L, unsigned int ncv) {
	// 1. Construct matrix operation object using the wrapper class SparseGenMatProd
	Spectra::SparseSymMatProd<float> l(L);

	// 2. Construct eigen solver object, requesting the maximum eigen value
	Spectra::SymEigsSolver<Spectra::SparseSymMatProd<float>> eigs(l, 1, ncv);

	// 3. Initialize and compute
	eigs.init();
	eigs.compute(Spectra::SortRule::LargestMagn);

	// 4. Retrieve results
	float max_eigen_value;
	if (eigs.info() == Spectra::CompInfo::Successful) {
		max_eigen_value = eigs.eigenvalues().real()(0);
	}

	if (max_eigen_value == 0) {
		std::cout << FRD("[ERROR]: ") << "Max Eigenvalue shouldn't be zero unless it's zero matrix." << std::endl;
		return EXIT_FAILURE;
	}

	return max_eigen_value;
}

int main(int argc, char** argv) {
	// load the mtx file
	Eigen::SparseMatrix<float> matrix;

	if (argc == 2) {
		Eigen::loadMarket(matrix, argv[1]);

		// Since, Eigen::loadMarket doesnot support symmetric matrix, we need to do further more steps.
		// Make sure the mtx file consists only lower part of the matrix which is a standard for symmetric matrix.
		int sym;
		bool iscomplex=false, isdense=false; // no-use, just for a parameter for Eigen::getMarketHeader
		Eigen::getMarketHeader(argv[1], sym, iscomplex, isdense);
		if (sym != 0) {
			Eigen::SparseMatrix<float> temp;
			temp = matrix;
			matrix = temp.selfadjointView<Eigen::Lower>();
		}

		if (matrix.nonZeros() == 0) {
			std::cout << FRD("[ERROR]: ") << "Matrix is empty." << std::endl;
			return EXIT_FAILURE;
		}

		// calculate max eigenvalue
		float max_eigen_value = calcMaxEigenvalue(matrix, std::min<unsigned int>(matrix.rows(), 20));
		std::cout << FGRN("[SUCCESS]: ") << "Max Eigenvalue: " << max_eigen_value << std::endl;

	} else {
		std::cout << FRD("[ERROR]: ") << "Please provide a path to a matrix market file." << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}