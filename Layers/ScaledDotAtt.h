#pragma once

#include "hls_stream.h"
#include "hls_vector.h"
#include "MatMul.h"
#include "SoftMax.h"

template<typename T, int sequence_length, int token_length>
void scaledotatt
(
	hls::stream<hls::vector<T, token_length>> &query,
	hls::stream<hls::vector<T, token_length>> &key,
	hls::stream<hls::vector<T, token_length>> &value,
	hls::stream<hls::vector<T, sequence_length>> &input_mask,
	hls::stream<hls::vector<T, token_length>> &result
)
{
	hls::stream<hls::vector<T, sequence_length>> softmax_att;

	#pragma HLS DATAFLOW
	matmul_scale_masked_softmax<T,sequence_length,token_length,sequence_length>(query, key, SCALE_FACTOR, input_mask, softmax_att);
	matmul<T, sequence_length, sequence_length, token_length>(softmax_att, value, result);
}
