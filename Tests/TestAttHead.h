#pragma once 
#include "TestUtils.h"
#include "AttHead.h"

template<typename T, int sequence_length, int token_length, int head_token_length>
void test_attHead(
	std::string* matIn_filename,
	std::string* weights_filename,
	std::string* biases_filename,
	std::string* matMask_filename,
	std::string* result_gold_filename,
	std::string* log_filename
) {
	T matIn[sequence_length][token_length]{};
	load_arr<T, sequence_length*token_length>((T*)matIn, matIn_filename);

	T matMask[sequence_length][sequence_length]{};
	load_arr<T, sequence_length*sequence_length>((T*)matMask, matMask_filename);

	T weights[num_linear_layers][token_length][head_token_length]{};
	load_arr<T, num_linear_layers*token_length*head_token_length>((T*)weights, weights_filename);

	T biases[num_linear_layers][head_token_length]{};
	load_arr<T, num_linear_layers*head_token_length>((T*)biases, biases_filename);

	T output[sequence_length][head_token_length]{};
	att_head<T, sequence_length, token_length, head_token_length>(matIn, matIn, matIn, matMask, weights, biases, output);
	compare_mat<T, sequence_length, head_token_length>(output, result_gold_filename, log_filename);
}