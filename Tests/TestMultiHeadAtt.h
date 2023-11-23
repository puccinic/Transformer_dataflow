#pragma once
#include "TestUtils.h"
#include "MultiheadAtt.h"

template<typename T, size_t num_heads, size_t sequence_length, size_t token_length, size_t head_token_length>
void test_multiheadatt(
	std::string* input_filename,
	std::string* mask_filename,
	std::string* weights_filename,
	std::string* biases_filename,
	std::string* linear_weights_filename,
	std::string* linear_bias_filename,
	std::string* result_gold_filename,
	std::string* log_filename
) {
	T input[sequence_length][token_length]{};
	load_arr<T, sequence_length*token_length>((T*)input, input_filename);

	T input_mask[sequence_length][sequence_length]{};
	load_arr<T, sequence_length*sequence_length>((T*)input_mask, mask_filename);

	
	T weights[num_heads][num_linear_layers][token_length][head_token_length]{};
	load_arr<T, num_heads*num_linear_layers*token_length*head_token_length>((T*)weights, weights_filename);

	T biases[num_heads][num_linear_layers][head_token_length]{};
	load_arr<T, num_heads*num_linear_layers*head_token_length>((T*)biases, biases_filename);

	T linear_weight[token_length][token_length]{};
	load_arr<T, token_length*token_length>((T*)linear_weight, linear_weights_filename);

	T linear_bias[token_length]{};
	load_arr<T, token_length>((T*)linear_bias, linear_bias_filename);

	T output[sequence_length][token_length]{};

	MultiHeadAtt<T, num_heads, sequence_length, token_length, head_token_length> mulitheadatt;
	mulitheadatt.init(weights, biases, linear_weight, linear_bias);
	mulitheadatt(input, input, input, input_mask, output);

	compare_mat<T, sequence_length, token_length>(output, result_gold_filename, log_filename);
}