#pragma once

#include "hls_stream.h"
#include "MultiHeadAtt.h"
#include "MatAdd.h"
#include "LayerNorm.h"
#include "FF.h"
#include "Synth_utils.h"

template<typename T, int num_heads, int sequence_length, int token_length, int head_token_length, int hidden>
void encoder(
	hls::stream<T> input[token_length],
	hls::stream<T> head_weights[num_heads][NUM_LINEAR_LAYERS][token_length],
	hls::stream<T> linear_weights[head_token_length*num_heads],
	hls::stream<T> linear_bias[token_length],
	hls::stream<T> ff_weights1[token_length],
	hls::stream<T> ff_biases1[hidden],
	hls::stream<T> ff_weights2[hidden],
	hls::stream<T> ff_biases2[token_length],
	hls::stream<T> gamma[NUM_LAYER_NORM][sequence_length],
	hls::stream<T> beta[NUM_LAYER_NORM][sequence_length],
	hls::stream<T> mean[NUM_LAYER_NORM][sequence_length],
    hls::stream<T> variance[NUM_LAYER_NORM][sequence_length],
	hls::stream<T> result[token_length]
)
{
	hls::stream<T, sequence_length> input_copy1[token_length]{};
	hls::stream<T, sequence_length> input_copy2[token_length]{};
	hls::stream<T, sequence_length> multi_head_result[token_length]{};
	hls::stream<T, sequence_length> matadd_result1[token_length]{};
	hls::stream<T, sequence_length> matadd_result1_copy1[token_length]{};
	hls::stream<T, sequence_length> matadd_result1_copy2[token_length]{};
	hls::stream<T, sequence_length> norm_result1[token_length]{};
	hls::stream<T, sequence_length> norm_result1_copy1[token_length]{};
	hls::stream<T, sequence_length> norm_result1_copy2[token_length]{};
	hls::stream<T, sequence_length> norm_result1_copy3[token_length]{};
	hls::stream<T, sequence_length> norm_result2[token_length]{};

	hls::stream<T, sequence_length> ff_result[token_length]{};
	hls::stream<T, sequence_length> matadd_result2[token_length]{};

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
