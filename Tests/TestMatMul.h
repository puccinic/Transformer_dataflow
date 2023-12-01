#pragma once
#include "TestUtils.h"
#include "MatMul.h"

template<typename T, int rows, int hidden, int cols>
void test_matmul(
	std::string* matA_filename,
	std::string* matB_filename,
	std::string* matResGold_filename,
	std::string* log_filename
) {
	T A[rows][hidden]{};
	load_arr<T, rows*hidden>((T*)A, matA_filename);

	T B[hidden][cols]{};
	load_arr<T, hidden*cols>((T*)B, matB_filename);

	T result[rows][cols]{};

	matmul<T, rows, hidden, cols>(A, B, result);

	compare_mat<T, rows, cols>(result, matResGold_filename, log_filename);
}
