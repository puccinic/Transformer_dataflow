#pragma once
#include "Linear.h"
#include "ScaledDotAtt.h"

template<typename T, size_t sequence_length, size_t token_length, size_t head_token_length>
void atthead(T query[sequence_length][token_length],
	T key[sequence_length][token_length],
	T value[sequence_length][token_length],
	T q_weights[token_length][head_token_length],
	T q_biases[head_token_length],
	T k_weights[token_length][head_token_length],
	T k_biases[head_token_length],
	T v_weights[token_length][head_token_length],
	T v_biases[head_token_length],
	T input_mask[sequence_length][sequence_length],
	T scale_factor,
	T result[sequence_length][head_token_length]) {
	
	T Q[sequence_length][head_token_length];
	linear<T, sequence_length, token_length, head_token_length>(query, q_weights, q_biases, Q);

	T K[sequence_length][head_token_length];
	linear<T, sequence_length, token_length, head_token_length>(key, k_weights, k_biases, K);

	T V[sequence_length][head_token_length];
	linear<T, sequence_length, token_length, head_token_length>(value, v_weights, v_biases, V);

	scaledotatt<T, sequence_length, head_token_length>(Q, K, V, input_mask, scale_factor, result);
}