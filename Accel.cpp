#include "ap_int.h"
typedef unsigned long size_t;
#define EPSILON 1e-5
int epsilon[2] = { 0, 0 };
constexpr size_t num_heads = 1;
constexpr size_t sequence_length = 10;
constexpr size_t token_length = 10;
constexpr size_t head_token_length = 10;
constexpr size_t hidden = 10;

#include "Encoder.h"

typedef int head_weights_t[num_heads][num_linear_layers][token_length][head_token_length];
typedef int head_biases_t[num_heads][num_linear_layers][head_token_length];

void accel(
	head_weights_t head_weights,
	head_biases_t head_biases,
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
