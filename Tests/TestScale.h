#pragma once
#include "TestUtils.h"
#include "Scale.h"

template<typename T, size_t rows, size_t cols>
void test_scale(std::string* input_filename,
	T scale_factor,
	std::string* result_gold_filename,
	std::string* log_filename) {

	T input[rows][cols]{};
	load_mat<T, rows, cols>(input, input_filename);

	T output[rows][cols]{};

	scale<T, rows, cols>(input, output, scale_factor);

	compare_mat<T, rows, cols>(output, result_gold_filename, log_filename);
}