#pragma once 
#include "AttHead.h"
#include "Concat.h"
#include "Linear.h"

template<typename T, int num_heads, int sequence_length, int token_length, int head_token_length>
void multi_head_att(
	T query[sequence_length][token_length],
	T key[sequence_length][token_length],
	T values[sequence_length][token_length],
	T input_mask[sequence_length][sequence_length],
	T head_weights[num_heads][num_linear_layers][token_length][head_token_length],
	T head_biases[num_heads][num_linear_layers][head_token_length],
	T linear_weights[token_length][token_length],
	T linear_bias[token_length],
	T result[sequence_length][token_length]
	) {
	T tmp1[num_heads][sequence_length][head_token_length];
multi_head_att_loop:
	for (int i = 0; i < num_heads; i++) {
		att_head<T, sequence_length, token_length, head_token_length>(
			query, 
			key, 
			values, input_mask, head_weights[i], 
			head_biases[i], 
			tmp1[i]
		);
	}
	T tmp2[sequence_length][token_length];
	concat_cols<T, sequence_length, head_token_length, num_heads>(tmp1, tmp2);
	linear<T, sequence_length, token_length, token_length>(tmp2, linear_weights, linear_bias, result);
}
