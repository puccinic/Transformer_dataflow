#pragma once
#include "TestUtils.h"
#include "Linear.h"

template<typename T, size_t rows, size_t hidden, size_t cols>
void test_linear(std::string* input_filename,
	std::string* weights_filename,
	std::string* biases_filename,
	std::string* result_gold_filename,
	std::string* log_filename) {

	T input[rows][hidden]{};
	load_arr<T, rows*hidden>((T*)input, input_filename);

	T weights[hidden][cols]{};
	load_arr<T, hidden*cols>((T*)weights, weights_filename);

	T biases[cols]{};
	load_arr<T, cols>(biases, biases_filename);

	T output[rows][cols]{};

	Linear<T, rows, hidden, cols> linear;
	linear.init(weights, biases);
	linear(input, output);
	compare_mat<T, rows, cols>(output, result_gold_filename, log_filename);
}