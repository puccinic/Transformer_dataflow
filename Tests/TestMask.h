#pragma once
#include "TestUtils.h"
#include "Mask.h"

template<typename T, int rows, int cols>
void test_mask(
	std::string* input_filename,
	std::string* mask_filename,
	std::string* result_gold_filename,
	std::string* log_filename
) {
	T input[rows][cols]{};
	load_arr<T, rows*cols>((T*)input, input_filename);

	T mask_mat[rows][cols]{};
	load_arr<T, rows*cols>((T*)mask_mat, mask_filename);
	
	T output[rows][cols]{};
	
	mask<T, rows, cols>(input, mask_mat, output);

	compare_mat<T, rows, cols>(output, result_gold_filename, log_filename);
}
