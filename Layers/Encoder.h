#pragma once
#include "MultiHeadAtt.h"
#include "MatAdd.h"
#include "LayerNorm.h"
#include "FF.h"
template<typename T, size_t num_heads, 
	size_t sequence_length, 
	size_t token_length, 
	size_t head_token_length,
	size_t hidden>
struct Encoder {
	
	MultiHeadAtt<T, num_heads, sequence_length, token_length, head_token_length>
		multi_head_att;
	LayerNorm<T, sequence_length, token_length> 
		layer_norm1, layer_norm2;
	FF<T, sequence_length, hidden, token_length> 
		ff;
	
	typedef T head_weights_t[num_heads][num_linear_layers][token_length][head_token_length];
	typedef T head_biases_t[num_heads][num_linear_layers][head_token_length];
	void init(head_biases_t head_weights, 
		head_biases_t head_biases,
		T linear_weights[token_length][token_length],
		T linear_bias[token_length],
		T ff_weights1[token_length][hidden],
		T ff_biases1[hidden],
		T ff_weights2[hidden][token_length],
		T ff_biases2[token_length],
		T epsilon[2][sequence_length],
		T gamma[2][sequence_length],
		T beta[2][sequence_length]) {
		multi_head_att.init(head_weights, head_biases, linear_weights, linear_bias);
		ff.init(ff_weights1, ff_biases1, ff_weights2, ff_biases2);
		layer_norm1.init(epsilon[0], gamma[0], beta[0]);
		layer_norm2.init(epsilon[1], gamma[1], beta[1]);
	}

	void operator()(T input[sequence_length][token_length],
		T input_mask[sequence_length][sequence_length],
		T result[sequence_length][sequence_length]) {

		T tmp1[sequence_length][token_length];
		multi_head_att(input, input, input, input_mask, tmp1);

		T tmp2[sequence_length][token_length];
		matadd<T, sequence_length, token_length>(input, tmp1, tmp2);
		
		T tmp3[sequence_length][token_length];
		layer_norm1(tmp2, tmp3);
		
		T tmp4[sequence_length][token_length];
		ff(tmp3, tmp4);

		T tmp5[sequence_length][token_length];
		matadd<T, sequence_length, token_length>(tmp3, tmp4, tmp5);

		layer_norm1(tmp5, result);
	}
};