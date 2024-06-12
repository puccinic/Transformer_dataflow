#pragma once

#include "hls_stream.h"
#include "MatMul.h"
#include "SoftMax.h"

template<typename T, int sequence_length, int token_length>
void scaledotatt(
	hls::stream<T> query[token_length],
	hls::stream<T> key[token_length],
	hls::stream<T> value[token_length],
	hls::stream<T> result[token_length]
)
{
	hls::stream<T, sequence_length> softmax_att[sequence_length]{};

	#pragma HLS DATAFLOW
	matmul_scale_masked_softmax<T,sequence_length,token_length,sequence_length>(query, key, SCALE_FACTOR, softmax_att);
	matmul<T, sequence_length, sequence_length, token_length>(softmax_att, value, result);
}
