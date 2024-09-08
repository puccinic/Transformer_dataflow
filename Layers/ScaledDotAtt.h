#pragma once

#include "hls_stream.h"
#include "MatMul.h"
#include "SoftMax.h"

template<typename iT, typename tmpT, typename rT, int sequence_length, int token_length>
void scaledotatt(
	hls::stream<iT>& query,
	hls::stream<iT>& key,
	hls::stream<iT>& value,
	hls::stream<rT>& result
)
{
	hls::stream<tmpT, sequence_length> softmax_att{};

	#pragma HLS DATAFLOW
	matmul_scale_masked_softmax<iT, iT, tmpT, sequence_length, token_length, sequence_length>(query, key, SCALE_FACTOR, softmax_att);
	matmul<tmpT, iT, rT, sequence_length, sequence_length, token_length>(softmax_att, value, result);
}
