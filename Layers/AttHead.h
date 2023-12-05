#pragma once
#include "Linear.h"
#include "ScaledDotAtt.h"

template<typename T, int sequence_length, int token_length, int head_token_length>
void att_head(
	T query[sequence_length][token_length],
	T key[sequence_length][token_length],
	T value[sequence_length][token_length],
	T input_mask[sequence_length][sequence_length],
	T weights[num_linear_layers][token_length][head_token_length],
	T biases[num_linear_layers][head_token_length],
	T result[sequence_length][head_token_length]
	) {

	T Q[sequence_length][head_token_length];
	linear<T, sequence_length, token_length, head_token_length>(query, weights[0], biases[0], Q);

	T K[sequence_length][head_token_length];
	linear<T, sequence_length, token_length, head_token_length>(key, weights[1], biases[1], K);

	T V[sequence_length][head_token_length];
	linear<T, sequence_length, token_length, head_token_length>(value, weights[2], biases[2], V);

	scaledotatt<T, sequence_length, head_token_length>(Q, K, V, input_mask, result);
}
