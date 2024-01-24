#pragma once
#include "MultiHeadAtt.h"
#include "MatAdd.h"
#include "LayerNorm.h"
#include "FF.h"
#include "Synth_utils.h"
template<typename T, int num_heads, int sequence_length, int token_length, int head_token_length, int hidden>
void encoder(
	T input[sequence_length][token_length],
	T input_mask[sequence_length][sequence_length],
	T head_weights[num_heads][NUM_LINEAR_LAYERS][token_length][head_token_length],
	T head_biases[num_heads][NUM_LINEAR_LAYERS][head_token_length],
	T linear_weights[token_length][token_length],
	T linear_bias[token_length],
	T ff_weights1[token_length][hidden],
	T ff_biases1[hidden],
	T ff_weights2[hidden][token_length],
	T ff_biases2[token_length],
	T epsilon[NUM_LAYER_NORM],
	T gamma[NUM_LAYER_NORM][token_length],
	T beta[NUM_LAYER_NORM][token_length],
	T result[sequence_length][token_length]
) {
	#pragma HLS DATAFLOW
	T input_copy1[sequence_length][token_length];
	T input_copy2[sequence_length][token_length];
	T input_copy3[sequence_length][token_length];
	T input_copy4[sequence_length][token_length];
	replicate4<T, sequence_length * token_length>((T*) input, (T*) input_copy1, (T*) input_copy2, (T*) input_copy3, (T*) input_copy4);

	T multi_head_result[sequence_length][token_length];
	multi_head_att<T, num_heads, sequence_length, token_length, head_token_length>(
		input_copy1, input_copy2, input_copy3,
		input_mask,
		head_weights,
		head_biases,
		linear_weights,
		linear_bias,
		multi_head_result
	);
	T matadd_result1[sequence_length][token_length];
	matadd<T, sequence_length, token_length>(input_copy4, multi_head_result, matadd_result1);

	T layer_norm_result[sequence_length][token_length];
	layer_norm<T, sequence_length, token_length>(matadd_result1, epsilon[0], gamma[0], beta[0], layer_norm_result);

	T layer_norm_result_copy1[sequence_length][token_length];
	T layer_norm_result_copy2[sequence_length][token_length];
	replicate2<T, sequence_length * token_length>((T*) layer_norm_result, (T*) layer_norm_result_copy1, (T*) layer_norm_result_copy2);

	T ff_result[sequence_length][token_length];
	ff<T, sequence_length, hidden, token_length>(layer_norm_result_copy1, ff_weights1, ff_biases1, ff_weights2, ff_biases2, ff_result);

	T matadd_result2[sequence_length][token_length];
	matadd<T, sequence_length, token_length>(layer_norm_result_copy2, ff_result, matadd_result2);

	layer_norm<T, sequence_length, token_length>(matadd_result2, epsilon[1], gamma[1], beta[1], result);
}
