#pragma once
#include "TestUtils.h"
#include "MatAdd.h"

template<typename T, size_t rows, size_t cols>
void test_matadd(
	std::string* matA_filename,
	std::string* matB_filename,
	std::string* result_gold_filename,
	std::string* log_filename
) {
	T matA[rows][cols]{};
	load_arr<T, rows*cols>((T*)matA, matA_filename);

	T matB[rows][cols]{};
	load_arr<T, rows*cols>((T*)matB, matB_filename);

	T result[rows][cols]{};

	matadd<T, rows, cols>(matA, matB, result);

	compare_mat<T, rows, cols>(result, result_gold_filename, log_filename);
}

