#pragma once
#include "TestUtils.h"
#include "LayerNorm.h"

template<typename T, int channels, int size>
void test_layernorm(
	T epsilon,
	std::string* input_filename,
	std::string* gamma_filename,
	std::string* beta_filename,
	std::string* vecResGold_filename,
	std::string* log_filename
) {
	T input[channels][size]{};
	load_arr<T, channels*size>((T*)input, input_filename);
	
	T gamma[channels]{};
	load_arr<T, channels>(gamma, gamma_filename);

	T beta[channels]{};
	load_arr<T, channels>(beta, beta_filename);

	T output[channels][size]{};
	layer_norm<T, channels, size>(input, epsilon, gamma, beta, output);
	compare_mat<T, channels, size>(output, vecResGold_filename, log_filename);
}

