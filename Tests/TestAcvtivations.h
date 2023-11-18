#pragma once
#include "TestUtils.h"
#include "Activations.h"

template<typename T, size_t rows, size_t cols>
void test_activations(std::string* input_filename,
	std::string* result_gold_filename,
	std::string* log_filename) {

	T input[rows][cols]{};
	load_arr<T, rows*cols>((T*)input, input_filename);

	T output[rows][cols]{};

	activation<T, rows, cols>(input, output);

	compare_mat<T, rows, cols>(output, result_gold_filename, log_filename);
}
