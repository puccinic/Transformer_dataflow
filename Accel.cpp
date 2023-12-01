#include "Encoder.h"
#define EPSILON 1e-5

int epsilon[2] = { 0, 0 };
constexpr int num_heads = 1;
constexpr int sequence_length = 10;
constexpr int token_length = 10;
constexpr int head_token_length = 10;
constexpr int hidden = 10;

#define ENCODER
#ifdef ENCODER
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
#endif

#ifdef MULTIHEAD
void accel(
	int head_weights[num_heads][num_linear_layers][token_length][head_token_length],
	int head_biases[num_heads][num_linear_layers][head_token_length],
	int linear_weights[token_length][token_length],
	int linear_bias[token_length],
	int ff_weights1[token_length][hidden],
	int ff_biases1[hidden],
	int ff_weights2[hidden][token_length],
	int ff_biases2[token_length],
	int input[sequence_length][token_length],
	int input_mask[sequence_length][sequence_length],
	int result[sequence_length][sequence_length]
) {
	multi_head_att<int, num_heads, sequence_length, token_length, head_token_length> (
		input,
		input,
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
		result
	);
}
#endif

#ifdef ATTHEAD
void accel(
	int weights[num_linear_layers][token_length][head_token_length],
	int biases[num_linear_layers][head_token_length],
	int result[sequence_length][token_length],
	int input_mask[sequence_length][sequence_length],
) {
	att_head<int, sequence_length, token_length, head_token_length>(
		input,
		input,
		input,
		input_mask,
		weights,
		biases,
		result
	);
}
#endif

#ifdef FDFRWRD
void accel(
	int weights1[sequence_length][hidden],
	int biases1[hidden],
	int weights2[hidden][token_length],
	int biases2[token_length],
	int input[sequence_length][token_length],
	int result[sequence_length][token_length],
) {
	ff<int, sequence_length, hidden, token_length>(
		input,
		weights1,
		biases1,
		weights2,
		biases2,
		result
	);
}
#endif

#ifdef LAYERNORM
void accel(
	int gamma[2][token_length],
	int beta[2][token_length],
	int input[sequence_length][token_length],
	int result[sequence_length][sequence_length]
) {
	layer_norm<int, sequence_length, token_length>(
		input,
		epsilon,
		gamma,
		beta,
		result
	);
}
#endif

#ifdef DOTPRODATT
void accel(
	int input[sequence_length][token_length],
	int input_mask[sequence_length][sequence_length],
	int result[sequence_length][token_length]
) {
	scaledotatt<int, sequence_length, token_length> (
		input, input, input,
		input_mask,
		result
	);
}
#endif

#ifdef LINEAR
void accel(
	int weights[hidden][token_length],
	int biases[token_length],
	int input[sequence_length][hidden],
	int result[sequence_length][token_length],
) {
	linear<int, sequence_length, hidden, token_length> (
		input,
		weights,
		biases,
		result
	);
}
#endif

#ifdef MATMUL
void accel(
	int matA[sequence_length][hidden],
	int matB[hidden][token_length],
	int matRes[sequence_length][token_length]
) {
	matmul<int, sequence_length, hidden, token_length> (
		matA,
		matB,
		matRes
	);
}
#endif

#ifdef SOFTMAX
void accel(int input[sequence_length], int result[sequence_length]) {
	softmax<int, sequence_length>(input, result);
}
#endif

#ifdef ACTIVATION
void accel(
	int input[sequence_length][token_length],
	int result[sequence_length][token_length]
) {
	activation<int, sequence_length, token_length>(
		input,
		result
	);
}
#endif