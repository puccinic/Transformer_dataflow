#pragma once

#include "hls_stream.h"
#include "MultiHeadAtt.h"
#include "MatAdd.h"
#include "LayerNorm.h"
#include "FF.h"
#include "Synth_utils.h"

template<
	typename input_T,
	typename attention_weight_T,
	typename linear_weight_T,
	typename linear_bias_T,
	typename feedforward_weight1_T,
	typename feedforward_weight2_T,
	typename feedforward_bias2_T,
	typename gamma_T,
	typename beta_T,
	typename norm_result1_T,
	typename attention_intermediate1_T,
	typename attention_intermediate2_T,
	typename attention_output_T,
	typename multi_head_attention_linear_intermediate_T,
	typename multi_head_attention_result_T,
	typename res_bock_T,
	typename norm_result2_T,
	typename feedforward_intermediate_T,
	typename feedforward_linear2_intermediate_T,
	typename feedforward_resutlt_T,
	typename result_T,
	int num_heads,
	int sequence_length,
	int token_length,
	int head_token_length,
	int hidden
>
void encoder(
	hls::stream<input_T> input[token_length],
	hls::stream<attention_weight_T> head_weights[num_heads][NUM_LINEAR_LAYERS][token_length],
	hls::stream<linear_weight_T> linear_weights[head_token_length*num_heads],
	hls::stream<linear_bias_T> linear_bias[token_length],
	hls::stream<feedforward_weight1_T> ff_weights1[token_length],
	hls::stream<feedforward_weight2_T> ff_weights2[hidden],
	hls::stream<feedforward_bias2_T> ff_biases2[token_length],
	hls::stream<gamma_T> gamma[NUM_LAYER_NORM][sequence_length],
	hls::stream<beta_T> beta[NUM_LAYER_NORM][sequence_length],
	hls::stream<result_T> result[token_length]
)
{
	hls::stream<input_T, sequence_length> input_copy1[token_length]{};
	hls::stream<input_T, sequence_length> input_copy2[token_length]{};
	hls::stream<norm_result1_T, sequence_length> norm_result1[token_length]{};
	hls::stream<norm_result1_T, sequence_length> norm_result1_copy1[token_length]{};
	hls::stream<norm_result1_T, sequence_length> norm_result1_copy2[token_length]{};
	hls::stream<norm_result1_T, sequence_length> norm_result1_copy3[token_length]{};
	hls::stream<multi_head_attention_result_T, sequence_length> multi_head_result[token_length]{};
	hls::stream<res_bock_T, sequence_length> matadd_result1[token_length]{};
	hls::stream<res_bock_T, sequence_length> matadd_result1_copy1[token_length]{};
	hls::stream<res_bock_T, sequence_length> matadd_result1_copy2[token_length]{};
	hls::stream<norm_result2_T, sequence_length> norm_result2[token_length]{};
	hls::stream<feedforward_resutlt_T, sequence_length> ff_result[token_length]{};

	#pragma HLS DATAFLOW
	replicate2<input_T, sequence_length, token_length>(input, input_copy1, input_copy2);

	opt_batch_norm<
		input_T,
		gamma_T,
		beta_T,
		norm_result1_T,
		sequence_length,
		token_length
	>(
		input_copy1,
		gamma[0],
		beta[0],
		norm_result1
	);

	replicate3<norm_result1_T, sequence_length, token_length>(
		norm_result1,
		norm_result1_copy1,
		norm_result1_copy2,
		norm_result1_copy3
	);

	multi_head_att<
		norm_result1_T,
		attention_weight_T,
		linear_weight_T,
		linear_bias_T,
		attention_intermediate1_T,
		attention_intermediate2_T,
		attention_output_T,
		multi_head_attention_linear_intermediate_T,
		multi_head_attention_result_T,
		num_heads,
		sequence_length,
		token_length,
		head_token_length
	>(
		norm_result1_copy1,
		norm_result1_copy2,
		norm_result1_copy3,
		head_weights,
		linear_weights,
		linear_bias,
		multi_head_result
	);

	matadd<input_T, multi_head_attention_result_T, res_bock_T, sequence_length, token_length>(
		input_copy2,
		multi_head_result,
		matadd_result1
	);

	replicate2<res_bock_T, sequence_length, token_length>(
		matadd_result1,
		matadd_result1_copy1,
		matadd_result1_copy2
	);

	opt_batch_norm<
		res_bock_T,
		gamma_T,
		beta_T,
		norm_result2_T,
		sequence_length,
		token_length
	>(
		matadd_result1_copy1,
		gamma[1],
		beta[1],
		norm_result2
	);

	ff<
		norm_result2_T,
		feedforward_weight1_T,
		feedforward_weight2_T,
		feedforward_bias2_T,
		feedforward_intermediate_T,
		feedforward_linear2_intermediate_T,
		feedforward_resutlt_T,
		sequence_length,
		hidden,
		token_length
	>(
		norm_result2,
		ff_weights1,
		ff_weights2,
		ff_biases2,
		ff_result
	);

	matadd<res_bock_T, feedforward_resutlt_T, result_T, sequence_length, token_length>(
		matadd_result1_copy2,
		ff_result,
		result
	);

}
