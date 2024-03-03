#pragma once

#include "hls_stream.h"
#include "hls_vector.h"
#include "MultiHeadAtt.h"
#include "MatAdd.h"
#include "LayerNorm.h"
#include "FF.h"
#include "Synth_utils.h"

template<typename T, int num_heads, int sequence_length, int token_length, int head_token_length, int hidden>
void encoder
(
	hls::stream<hls::vector<T, token_length>> &input,
	hls::stream<hls::vector<T, sequence_length>> &input_mask,
	hls::stream<hls::vector<T, token_length>> head_weights[num_heads][NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<T, head_token_length>> head_biases[num_heads][NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<T, token_length>> &linear_weights,
	hls::stream<hls::vector<T, token_length>> &linear_bias,
	hls::stream<hls::vector<T, token_length>> &ff_weights1,
	hls::stream<hls::vector<T, hidden>> &ff_biases1,
	hls::stream<hls::vector<T, hidden>> &ff_weights2,
	hls::stream<hls::vector<T, token_length>> &ff_biases2,
	T epsilon[NUM_LAYER_NORM],
	hls::stream<hls::vector<T, token_length>> gamma[NUM_LAYER_NORM],
	hls::stream<hls::vector<T, token_length>> beta[NUM_LAYER_NORM],
#if defined(USING_BATCH_NORM)
	hls::stream<hls::vector<T, token_length>> mean[NUM_LAYER_NORM],
    hls::stream<hls::vector<T, token_length>> stddev[NUM_LAYER_NORM],
#endif /* using batch norm */
	hls::stream<hls::vector<T, token_length>> &result
)
{
	hls::stream<hls::vector<T, token_length>> input_copy1("input_copy1");
	hls::stream<hls::vector<T, token_length>> input_copy2("input_copy2");
	hls::stream<hls::vector<T, token_length>> input_copy3("input_copy3");
	hls::stream<hls::vector<T, token_length>> input_copy4("input_copy4");
	hls::stream<hls::vector<T, token_length>> multi_head_result("multi_head_res");
	hls::stream<hls::vector<T, token_length>> matadd_result1("matadd_res1");
	hls::stream<hls::vector<T, token_length>> norm_result("norm_res");
	hls::stream<hls::vector<T, token_length>> norm_result_copy1("norm_res_copy1");
	hls::stream<hls::vector<T, token_length>> norm_result_copy2("norm_res_copy2");
	hls::stream<hls::vector<T, token_length>> ff_result("ff_res");
	hls::stream<hls::vector<T, token_length>> matadd_result2("matadd_res2");

	#pragma HLS DATAFLOW
	replicate4<T, sequence_length, token_length>(input, input_copy1, input_copy2, input_copy3, input_copy4);
	multi_head_att<T, num_heads, sequence_length, token_length, head_token_length>
	(
		input_copy1, input_copy2, input_copy3,
		input_mask,
		head_weights,
		head_biases,
		linear_weights,
		linear_bias,
		multi_head_result
	);

	matadd<T, sequence_length, token_length>(input_copy4, multi_head_result, matadd_result1);

#if defined(USING_BATCH_NORM)
	batch_norm<T, sequence_length, token_length>(matadd_result1, epsilon[0], gamma[0], beta[0], mean[0], stddev[0], norm_result);
#else
	layer_norm<T, sequence_length, token_length>(matadd_result1, epsilon[0], gamma[0], beta[0], norm_result);
#endif /* using batch norm */

	replicate2<T, sequence_length, token_length>(norm_result, norm_result_copy1, norm_result_copy2);

	ff<T, sequence_length, hidden, token_length>(norm_result_copy1, ff_weights1, ff_biases1, ff_weights2, ff_biases2, ff_result);

	matadd<T, sequence_length, token_length>(norm_result_copy2, ff_result, matadd_result2);

#if defined(USING_BATCH_NORM)
	batch_norm<T, sequence_length, token_length>(matadd_result2, epsilon[1], gamma[1], beta[1], mean[1], stddev[1], result);
#else
	layer_norm<T, sequence_length, token_length>(matadd_result2, epsilon[1], gamma[1], beta[1], result);
#endif /* using batch norm */
}
