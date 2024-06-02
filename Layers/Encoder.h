#pragma once

#include "hls_stream.h"
#include "hls_vector.h"
#include "MultiHeadAtt.h"
#include "MatAdd.h"
#include "LayerNorm.h"
#include "FF.h"
#include "Synth_utils.h"

template<typename T, int num_heads, int sequence_length, int token_length, int head_token_length, int hidden>
void encoder(
	hls::stream<hls::vector<T, token_length>> &input,
	hls::stream<hls::vector<T, token_length>> head_weights[num_heads][NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<T, head_token_length*num_heads>> &linear_weights,
	hls::stream<hls::vector<T, token_length>> &linear_bias,
	hls::stream<hls::vector<T, token_length>> &ff_weights1,
	hls::stream<hls::vector<T, hidden>> &ff_biases1,
	hls::stream<hls::vector<T, hidden>> &ff_weights2,
	hls::stream<hls::vector<T, token_length>> &ff_biases2,
	hls::stream<hls::vector<T, sequence_length>> gamma[NUM_LAYER_NORM],
	hls::stream<hls::vector<T, sequence_length>> beta[NUM_LAYER_NORM],
	hls::stream<hls::vector<T, sequence_length>> mean[NUM_LAYER_NORM],
    hls::stream<hls::vector<T, sequence_length>> variance[NUM_LAYER_NORM],
	hls::stream<hls::vector<T, token_length>> &result
)
{
	hls::stream<hls::vector<T, token_length>, sequence_length> input_copy1("input_copy1");
	hls::stream<hls::vector<T, token_length>, sequence_length> input_copy2("input_copy2");
	hls::stream<hls::vector<T, token_length>, sequence_length> multi_head_result("multi_head_res");
	hls::stream<hls::vector<T, token_length>, sequence_length> matadd_result1("matadd_res1");
	hls::stream<hls::vector<T, token_length>, sequence_length> matadd_result1_copy1("matadd_res1_copy1");
	hls::stream<hls::vector<T, token_length>, sequence_length> matadd_result1_copy2("matadd_res1_copy2");
	hls::stream<hls::vector<T, token_length>, sequence_length> norm_result1("norm_res1");
	hls::stream<hls::vector<T, token_length>, sequence_length> norm_result1_copy1("norm_res_copy1");
	hls::stream<hls::vector<T, token_length>, sequence_length> norm_result1_copy2("norm_res_copy2");
	hls::stream<hls::vector<T, token_length>, sequence_length> norm_result1_copy3("norm_res_copy3");
	hls::stream<hls::vector<T, token_length>, sequence_length> norm_result2("norm_res2");

	hls::stream<hls::vector<T, token_length>, sequence_length> ff_result("ff_res");
	hls::stream<hls::vector<T, token_length>, sequence_length> matadd_result2("matadd_res2");

	#pragma HLS DATAFLOW
	replicate2<T, sequence_length, token_length>(input, input_copy1, input_copy2);

	batch_norm<T, sequence_length, token_length>(
		input_copy1,
		gamma[0],
		beta[0],
		mean[0],
		variance[0],
		norm_result1
	);

	replicate3<T, sequence_length, token_length>(
		norm_result1,
		norm_result1_copy1,
		norm_result1_copy2,
		norm_result1_copy3
	);

	multi_head_att<T, num_heads, sequence_length, token_length, head_token_length>(
		norm_result1_copy1,
		norm_result1_copy2,
		norm_result1_copy3,
		head_weights,
		linear_weights,
		linear_bias,
		multi_head_result
	);

	matadd<T, sequence_length, token_length>(
		input_copy2,
		multi_head_result,
		matadd_result1
	);

	replicate2<T, sequence_length, token_length>(
		matadd_result1,
		matadd_result1_copy1,
		matadd_result1_copy2
	);

	batch_norm<T, sequence_length, token_length>(
		matadd_result1_copy1,
		gamma[1],
		beta[1],
		mean[1],
		variance[1],
		norm_result2
	);

	ff<T, sequence_length, hidden, token_length>(
		norm_result2,
		ff_weights1,
		ff_biases1,
		ff_weights2,
		ff_biases2,
		ff_result
	);

	matadd<T, sequence_length, token_length>(
		matadd_result1_copy2,
		ff_result,
		result
	);

}
