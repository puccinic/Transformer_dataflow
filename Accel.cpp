#include "Encoder.h"

#define EPSILON 1e-5
double epsilon[2] = { EPSILON, EPSILON };
constexpr size_t num_heads = 10;
constexpr size_t sequence_length = 10;
constexpr size_t token_length = 10;
constexpr size_t head_token_length = 10;
constexpr size_t hidden = 10;

typedef short int head_weights_t[num_heads][num_linear_layers][token_length][head_token_length];
typedef short int head_biases_t[num_heads][num_linear_layers][head_token_length];

void accel(
	head_weights_t head_weights,
	head_biases_t head_biases,
	short int linear_weights[token_length][token_length],
	short int linear_bias[token_length],
	short int ff_weights1[token_length][hidden],
	short int ff_biases1[hidden],
	short int ff_weights2[hidden][token_length],
	short int ff_biases2[token_length],
	double gamma[2][token_length],
	double beta[2][token_length],
	short int input[sequence_length][token_length],
	short int input_mask[sequence_length][sequence_length],
	short int result[sequence_length][sequence_length]
) {
	Encoder<short int, num_heads, sequence_length, token_length, head_token_length, hidden> encoder;
	encoder.init(
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
		beta
	);
	encoder(input, input_mask, result);
}

