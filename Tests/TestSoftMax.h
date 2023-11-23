#pragma once
#include "TestUtils.h"
#include "SoftMax.h"

template<typename T, size_t size>
void test_softmax(
	std::string* input_filename,
	std::string* vecResGold_filename,
	std::string* log_filename
) {
	T input[size]{};
	load_arr<T, size>(input, input_filename);

	T result[size]{};

	softmax<T, size>(input, result);

	compare_vec<T, size>(result, vecResGold_filename, log_filename);
}

