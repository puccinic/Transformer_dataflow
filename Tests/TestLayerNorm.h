#pragma once
#include "TestUtils.h"
#include "LayerNorm.h"

template<typename T, size_t size>
void test_layernorm(std::string* input_filename,
	std::string* vecResGold_filename,
	std::string* log_filename) {

	T input[size]{};
	load_arr<T, size>(input, input_filename);

	T result[size]{};


}

