#pragma once
#include "TestUtils.h"
#include "FF.h"

template<typename T, size_t rows, size_t hidden, size_t cols>
void test_linear(std::string* input_filename,
	std::string* weight1_filename,
	std::string* biases1_filename,
	std::string* weight2_filename,
	std::string* biases2_filename,
	std::string* result_gold_filename,
	std::string* log_filename) {

	T input[rows][cols]{};
	load_mat<T, rows, cols>(input, input_filename);

	T weights1[cols][hidden]{};
	load_mat<T, cols, hidden>(weights1, weight1_filename);

	T biases1[hidden]{};
	load_arr<T, hidden>(biases1, biases1_filename);

	T weights2[hidden][cols]{};
	load_mat<T, hidden, cols>(weights2, weight2_filename);

	T biases2[cols]{};
	load_arr<T, cols>(biases2, biases2_filename);

	T output[rows][cols];

	FF<T, rows, hidden, cols>(input, weights1, biases1, weights2, biases2, output);
	
	compare_mat<T, rows, cols>(output, result_gold_filename, log_filename);
}