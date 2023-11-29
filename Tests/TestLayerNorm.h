#pragma once
#include "TestUtils.h"
#include "LayerNorm.h"

template<typename T, size_t channels, size_t size>
void test_layernorm(
	double epsilon,
	std::string* input_filename,
	std::string* gamma_filename,
	std::string* beta_filename,
	std::string* vecResGold_filename,
	std::string* log_filename
) {
	T input[channels][size]{};
	load_arr<T, channels*size>((T*)input, input_filename);
	
	double gamma[channels]{};
	load_arr<double, channels>(gamma, gamma_filename);

	double beta[channels]{};
	load_arr<double, channels>(beta, beta_filename);

	T output[channels][size]{};
	layer_norm<T, channels, size>(input, epsilon, gamma, beta, output);
	compare_mat<T, channels, size>(output, vecResGold_filename, log_filename);
}

