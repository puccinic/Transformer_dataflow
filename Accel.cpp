#include <hls_math.h>
#include <ap_fixed.h>
#include "Encoder.h"
#define EPSILON 1e-5
typedef ap_fixed<8,4,AP_RND> idata_t;
typedef ap_fixed<8,4,AP_RND> odata_t;
idata_t epsilon[2] = { 0, 0 };
constexpr int num_heads = 1;
constexpr int sequence_length = 10;
constexpr int token_length = 10;
constexpr int head_token_length = 10;
constexpr int hidden = 10;

#define ENCODER
#ifdef ENCODER
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
) {
	encoder<idata_t, num_heads, sequence_length, token_length, head_token_length, hidden>(
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
	idata_t head_weights[num_heads][num_linear_layers][token_length][head_token_length],
	idata_t head_biases[num_heads][num_linear_layers][head_token_length],
	idata_t linear_weights[token_length][token_length],
	idata_t linear_bias[token_length],
	idata_t input[sequence_length][token_length],
	idata_t input_mask[sequence_length][sequence_length],
	odata_t result[sequence_length][sequence_length]
) {
	multi_head_att<idata_t, num_heads, sequence_length, token_length, head_token_length> (
		input,
		input,
		input,
		input_mask,
		head_weights,
		head_biases,
		linear_weights,
		linear_bias,
		result
	);
}
#endif

#ifdef ATTHEAD
void accel(
	idata_t weights[num_linear_layers][token_length][head_token_length],
	idata_t biases[num_linear_layers][head_token_length],
	idata_t input[sequence_length][token_length],
	idata_t input_mask[sequence_length][sequence_length],
	odata_t result[sequence_length][token_length],
) {
	att_head<idata_t, sequence_length, token_length, head_token_length>(
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
	idata_t weights1[sequence_length][hidden],
	idata_t biases1[hidden],
	idata_t weights2[hidden][token_length],
	idata_t biases2[token_length],
	idata_t input[sequence_length][token_length],
	odata_t result[sequence_length][token_length]
) {
	ff<idata_t, sequence_length, hidden, token_length>(
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
	idata_t gamma[2][token_length],
	idata_t beta[2][token_length],
	idata_t input[sequence_length][token_length],
	odata_t result[sequence_length][sequence_length]
) {
	layer_norm<idata_t, sequence_length, token_length>(
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
	idata_t input[sequence_length][token_length],
	idata_t input_mask[sequence_length][sequence_length],
	odata_t result[sequence_length][token_length]
) {
	scaledotatt<idata_t, sequence_length, token_length> (
		input, input, input,
		input_mask,
		result
	);
}
#endif

#ifdef LINEAR
void accel(
	idata_t weights[hidden][token_length],
	idata_t biases[token_length],
	idata_t input[sequence_length][hidden],
	odata_t result[sequence_length][token_length]
) {
	linear<idata_t, sequence_length, hidden, token_length> (
		input,
		weights,
		biases,
		result
	);
}
#endif

#ifdef MATMUL
void accel(
	idata_t matA[sequence_length][hidden],
	idata_t matB[hidden][token_length],
	odata_t matRes[sequence_length][token_length]
) {
	matmul<idata_t, sequence_length, hidden, token_length> (
		matA,
		matB,
		matRes
	);
}
#endif

#ifdef SOFTMAX
void accel(idata_t input[sequence_length], idata_t result[sequence_length]) {
	softmax<idata_t, sequence_length>(input, result);
}
#endif

#ifdef ACTIVATION
void accel(
	idata_t input[sequence_length][token_length],
	odata_t result[sequence_length][token_length]
) {
	activation<idata_t, sequence_length, token_length>(
		input,
		result
	);
}
#endif
