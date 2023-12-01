#include "Encoder.h"
#define EPSILON 1e-5

int epsilon[2] = { 0, 0 };
constexpr int num_heads = 1;
constexpr int sequence_length = 10;
constexpr int token_length = 10;
constexpr int head_token_length = 10;
constexpr int hidden = 10;


void accel(
	int head_weights[num_heads][num_linear_layers][token_length][head_token_length],
	int head_biases[num_heads][num_linear_layers][head_token_length],
	int linear_weights[token_length][token_length],
	int linear_bias[token_length],
	int ff_weights1[token_length][hidden],
	int ff_biases1[hidden],
	int ff_weights2[hidden][token_length],
	int ff_biases2[token_length],
	int gamma[2][token_length],
	int beta[2][token_length],
	int input[sequence_length][token_length],
	int input_mask[sequence_length][sequence_length],
	int result[sequence_length][sequence_length]
) {
	encoder<int, num_heads, sequence_length, token_length, head_token_length, hidden>(
		input, 
		input_mask,
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
		result
	);
}
