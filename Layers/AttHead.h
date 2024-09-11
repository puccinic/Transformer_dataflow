#pragma once

#include "hls_stream.h"
#include "Linear.h"
#include "ScaledDotAtt.h"
#include "Synth_utils.h"

template<typename iT, typename wT, typename tmp1T, typename tmp2T, typename rT, int sequence_length, int token_length, int head_token_length>
void att_head(
	hls::stream<iT>& input,
	hls::stream<wT>& weights,
	hls::stream<rT>& result
)
{
	hls::stream<tmp1T, sequence_length*head_token_length*3> qkv{};
	hls::stream<tmp1T, sequence_length*head_token_length> Q{};
	hls::stream<tmp1T, sequence_length*head_token_length> K{};
	hls::stream<tmp1T, sequence_length*head_token_length> V{};

	#pragma HLS DATAFLOW
	matmul_transpose<iT, wT, tmp1T, sequence_length, token_length, head_token_length*3>(input, weights, qkv);
	split3<tmp1T, sequence_length, head_token_length>(qkv, Q, K, V);
	scaledotatt<tmp1T, tmp2T, rT, sequence_length, head_token_length>(Q, K, V, result);
}
