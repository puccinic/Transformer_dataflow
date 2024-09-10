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
	hls::stream<wT, token_length*sequence_length> input_copy1{};
	hls::stream<wT, token_length*sequence_length> input_copy2{};
	hls::stream<wT, token_length*sequence_length> input_copy3{};
	hls::stream<wT, token_length*head_token_length> q_weights{};
	hls::stream<wT, token_length*head_token_length> k_weights{};
	hls::stream<wT, token_length*head_token_length> v_weights{};
	hls::stream<tmp1T, sequence_length*token_length> Q{};
	hls::stream<tmp1T, sequence_length*token_length> K{};
	hls::stream<tmp1T, sequence_length*token_length> V{};

	#pragma HLS DATAFLOW
	replicate3<input_T, sequence_length, token_length>(input, input_copy1, input_copy2, input_copy3);
	split3<wT, head_token_length, token_length>(weights, q_weights, k_weights, v_weights);
	matmul_transpose<iT, wT, tmp1T, sequence_length, token_length, head_token_length>(input_copy1, q_weights, Q);
	matmul_transpose<iT, wT, tmp1T, sequence_length, token_length, head_token_length>(input_copy2, k_weights, K);
	matmul_transpose<iT, wT, tmp1T, sequence_length, token_length, head_token_length>(input_copy3, v_weights, V);
	scaledotatt<tmp1T, tmp2T, rT, sequence_length, head_token_length>(Q, K, V, result);
}
