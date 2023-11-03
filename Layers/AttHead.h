#pragma once
#include "Linear.h"
#include "ScaledDotAtt.h"



template<typename T, size_t sequence_length, size_t token_length, size_t head_token_length, T scale_factor>
struct AttHead {
	constexpr size_t num_linear_layers = 3;
	Linear<T, sequence_length, token_length, token_length> linear_q, linear_k, linear_v;
	
	void init(T weights[num_linear_layers][token_length][head_token_length],
		T biases[num_linear_layers][head_token_length]) {
		linear_q.init(weights[0], biases[0]);
		linear_k.init(weights[1], biases[1]);
		linear_v.init(weights[2], biases[2]);
	}

	void operator()(T query[sequence_length][token_length],
		T key[sequence_length][token_length],
		T value[sequence_length][token_length],
		T input_mask[sequence_length][sequence_length],
		T result[sequence_length][head_token_length]) {

		T Q[sequence_length][head_token_length]{};
		linear_q(query, Q);

		T K[sequence_length][head_token_length]{};
		linear_k(key, K);

		T V[sequence_length][head_token_length]{};
		linear_v(value, V);

		scaledotatt<T, sequence_length, head_token_length, scale_factor>(Q, K, V, input_mask, result);
	}
};