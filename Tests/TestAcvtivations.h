#pragma once
#include "TestUtils.h"
#include "Activations.h"

template<typename T, size_t rows, size_t cols>
void test_activations(std::string* input_filename,
	std::string* result_gold_filename,
	std::string* log_filename,
	std::function<T(T)> activation_func) {

	T input[rows][cols]{};
	load_mat<T, rows, cols>(input, input_filename);

	T output[rows][cols]{};

	activation<T, rows, cols>(input, output, activation_func);

	compare_mat<T, rows, cols>(output, result_gold_filename, log_filename);
}
