#pragma once
#include "MultiHeadAtt.h"
#include "MatAdd.h"
#include "LayerNorm.h"
#include "FF.h"
template<typename T, int num_heads, int sequence_length, int token_length, int head_token_length, int hidden>
void encoder(
	T input[sequence_length][token_length],
	T input_mask[sequence_length][sequence_length],
	T head_weights[num_heads][num_linear_layers][token_length][head_token_length],
	T head_biases[num_heads][num_linear_layers][head_token_length],
	T linear_weights[token_length][token_length],
	T linear_bias[token_length],
	T ff_weights1[token_length][hidden],
	T ff_biases1[hidden],
	T ff_weights2[hidden][token_length],
	T ff_biases2[token_length],
	T epsilon[2],
	T gamma[2][token_length],
	T beta[2][token_length],
	T result[sequence_length][sequence_length]
	) {

	T tmp1[sequence_length][token_length];
	multi_head_att<T, num_heads, sequence_length, token_length, head_token_length>(
		input, input, input, 
		input_mask, 
		head_weights, 
		head_biases, 
		linear_weights, 
		linear_bias, 
		tmp1
	);

	T tmp2[sequence_length][token_length];
	matadd<T, sequence_length, token_length>(input, tmp1, tmp2);

	T tmp3[sequence_length][token_length];
	layer_norm<T, sequence_length, token_length>(tmp2, epsilon[0], gamma[0], beta[0], tmp3);

	T tmp4[sequence_length][token_length];
	ff<T, sequence_length, hidden, token_length>(tmp3, ff_weights1, ff_biases1, ff_weights2, ff_biases2, tmp4);

	T tmp5[sequence_length][token_length];
	matadd<T, sequence_length, token_length>(tmp3, tmp4, tmp5);

	layer_norm<T, sequence_length, token_length>(tmp5, epsilon[1], gamma[1], beta[1], result);
}
