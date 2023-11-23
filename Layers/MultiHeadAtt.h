#pragma once 
#include "AttHead.h"
#include "Concat.h"
#include "Linear.h"

template<typename T, size_t num_heads, size_t sequence_length, size_t token_length, size_t head_token_length>
struct MultiHeadAtt {
	AttHead<T, sequence_length, token_length, head_token_length> heads[num_heads];
	Linear<T, sequence_length, token_length, token_length> linear;

	
	typedef T head_weights_t[num_heads][num_linear_layers][token_length][head_token_length];
	typedef T head_biases_t[num_heads][num_linear_layers][head_token_length];
	
	void init(
		head_weights_t head_weights,
		head_biases_t head_biases,
		T linear_weights[token_length][token_length],
		T linear_bias[token_length]
	) {
		for (size_t i = 0; i < num_heads; i++) {
			heads[i].init(head_weights[i], head_biases[i]);
		}
		linear.init(linear_weights, linear_bias);
	}

	void operator()(
		T query[sequence_length][token_length],
		T key[sequence_length][token_length],
		T values[sequence_length][token_length],
		T input_mask[sequence_length][sequence_length],
		T result[sequence_length][token_length]
		) {
		T tmp1[num_heads][sequence_length][head_token_length]{};
		for (size_t i = 0; i < num_heads; i++) {
			heads[i](query, key, values, input_mask, tmp1[i]);
		}
		T tmp2[sequence_length][token_length]{};
		concat_cols<T, sequence_length, head_token_length, num_heads>(tmp1, tmp2);

		linear(tmp2, result);
	}
};
