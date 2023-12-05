#pragma once
#include <ap_fixed.h>
#define EPSILON 1e-5
typedef int idata_t;
typedef int odata_t;
constexpr int num_heads = 1;
constexpr int sequence_length = 10;
constexpr int token_length = 10;
constexpr int head_token_length = 10;
constexpr int hidden = 10;
constexpr int num_linear_layers = 3;
void accel(
	idata_t head_weights[num_heads][num_linear_layers][token_length][head_token_length],
	idata_t head_biases[num_heads][num_linear_layers][head_token_length],
	idata_t linear_weights[token_length][token_length],
	idata_t linear_bias[token_length],
	idata_t ff_weights1[token_length][hidden],
	idata_t ff_biases1[hidden],
	idata_t ff_weights2[hidden][token_length],
	idata_t ff_biases2[token_length],
	idata_t gamma[2][token_length],
	idata_t beta[2][token_length],
	idata_t input[sequence_length][token_length],
	idata_t input_mask[sequence_length][sequence_length],
	odata_t result[sequence_length][token_length]
);

