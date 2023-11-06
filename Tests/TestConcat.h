#include "TestUtils.h"
#include "Concat.h"

template<typename T, size_t rows, size_t cols>
void test_concat(std::string* matA_filename,
	std::string* matB_filename,
	std::string* result_gold_filename,
	std::string* log_filename) {

	T matA[rows][cols]{};
	load_mat<T, rows, cols>(matA, matA_filename);

	T matB[rows][cols]{};
	load_mat<T, rows, cols>(matB, matB_filename);

	T result[rows][cols*2]{};

	T (*matrices[2])[rows][cols];
	matrices[0] = matA;
	matrices[1] = matB;
	
	concat_cols<T, rows, cols, 2>(matrices, result);
	
	compare_mat<T, rows, 2*cols>(result, result_gold_filename, log_filename);
}