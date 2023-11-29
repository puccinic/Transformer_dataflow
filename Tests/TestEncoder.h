#pragma once
#include "TestUtils.h"
#include "Encoder.h"

template<typename T, size_t num_heads,
	size_t sequence_length,
	size_t token_length,
	size_t head_token_length,
	size_t hidden>
	void test_encoder( 
		std::string* matIn_filename, 
		std::string* matMask_filename,
		std::string* headWeights_filename,
		std::string* headBiases_filename,
		std::string* attWeights_filename,
		std::string* attBiases_filename,
		std::string* ffWeights1_filename,
		std::string* ffBiases1_filename,
		std::string* ffWeights2_filename,
		std::string* ffBiases2_filename,
		double epsilon[2],
		std::string* gamma_filename,
		std::string* beta_filename,
		std::string* result_gold_filename,
		std::string* log_filename
	) {
	T input[sequence_length][token_length]{};
	load_arr<T, sequence_length*token_length>((T*)input, matIn_filename);
		
	T input_mask[sequence_length][sequence_length]{};
	load_arr<T, sequence_length*sequence_length>((T*)input_mask, matMask_filename);

	T head_weights[num_heads][num_linear_layers][token_length][head_token_length]{};
	load_arr<T, num_heads*num_linear_layers*token_length*head_token_length>((T*)head_weights, headWeights_filename);

	T head_biases[num_heads][num_linear_layers][head_token_length]{};
	load_arr<T, num_heads*num_linear_layers*head_token_length>((T*)head_biases, headBiases_filename);

	T linear_weights[token_length][token_length]{};
	load_arr<T, token_length*token_length>((T*)linear_weights, attWeights_filename);

	T linear_bias[token_length]{};
	load_arr<T, token_length>((T*)linear_bias, attBiases_filename);

	T ff_weights1[token_length][hidden]{};
	load_arr<T, token_length*hidden>((T*)ff_weights1, ffWeights1_filename);

	T ff_biases1[hidden]{};
	load_arr<T, hidden>(ff_biases1, ffBiases1_filename);

	T ff_weights2[hidden][token_length]{};
	load_arr<T, hidden*token_length>((T*)ff_weights2, ffWeights2_filename);

	T ff_biases2[token_length]{};
	load_arr<T, token_length>(ff_biases2, ffBiases2_filename);

	double gamma[2][token_length]{};
	load_arr<double, 2*token_length>((double*)gamma, gamma_filename);
		
	double beta[2][token_length]{};
	load_arr<double, 2*token_length>((double*)beta, beta_filename);

	T output[sequence_length][sequence_length]{};
	encoder<T, num_heads, sequence_length, token_length, head_token_length, hidden>
		(
			input, input_mask,
			head_weights,
			head_biases,
			linear_weights,
			linear_bias,
			ff_weights1,
			ff_biases1,
			ff_weights2,
			ff_biases2,
			epsilon,
			gamma,
			beta,
			output
		);
	compare_mat<T, sequence_length, token_length>(output, result_gold_filename, log_filename);
}