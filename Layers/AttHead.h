#pragma once
#include "Linear.h"
#include "ScaledDotAtt.h"
#include "Synth_utils.h"

template<typename T, int sequence_length, int token_length, int head_token_length>
void att_head(
	T query[sequence_length][token_length],
	T key[sequence_length][token_length],
	T value[sequence_length][token_length],
	T input_mask[sequence_length][sequence_length],
	T weights[NUM_LINEAR_LAYERS][token_length][head_token_length],
	T biases[NUM_LINEAR_LAYERS][head_token_length],
	T result[sequence_length][head_token_length]
) {
	//#pragma HLS DATAFLOW
	T q_weights[token_length][head_token_length];
	T k_weights[token_length][head_token_length];
	T v_weights[token_length][head_token_length];
	split3<T, token_length * head_token_length>( (T (*)[token_length * head_token_length]) weights, (T*) q_weights, (T*) k_weights, (T*) v_weights);

	T q_biases[head_token_length];
	T k_biases[head_token_length];
	T v_biases[head_token_length];
	split3<T, head_token_length>(biases, q_biases, k_biases, v_biases);

	T Q[sequence_length][head_token_length];
	linear<T, sequence_length, token_length, head_token_length>(query, q_weights, q_biases, Q);

	T K[sequence_length][head_token_length];
	linear<T, sequence_length, token_length, head_token_length>(key, k_weights, k_biases, K);

	T V[sequence_length][head_token_length];
	linear<T, sequence_length, token_length, head_token_length>(value, v_weights, v_biases, V);

	scaledotatt<T, sequence_length, head_token_length>(Q, K, V, input_mask, result);
}
