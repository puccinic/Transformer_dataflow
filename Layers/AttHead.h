#pragma once
#include "Linear.h"
#include "ScaledDotAtt.h"

constexpr size_t num_linear_layers = 3;

template<typename T, size_t sequence_length, size_t token_length, size_t head_token_length>
struct AttHead {
	
	Linear<T, sequence_length, token_length, head_token_length> linear_q, linear_k, linear_v;
	typedef T weights_t[num_linear_layers][token_length][head_token_length];
	typedef T biases_t[num_linear_layers][head_token_length];

	void init(weights_t weights, biases_t biases) {
		linear_q.init(weights[0], biases[0]);
		linear_k.init(weights[1], biases[1]);
		linear_v.init(weights[2], biases[2]);
	}

	void operator() (
		T query[sequence_length][token_length],
		T key[sequence_length][token_length],
		T value[sequence_length][token_length],
		T input_mask[sequence_length][sequence_length],
		T result[sequence_length][head_token_length]
		) {

		T Q[sequence_length][head_token_length]{};
		linear_q(query, Q);

		T K[sequence_length][head_token_length]{};
		linear_k(key, K);

		T V[sequence_length][head_token_length]{};
		linear_v(value, V);

		scaledotatt<T, sequence_length, head_token_length>(Q, K, V, input_mask, result);
	}
};