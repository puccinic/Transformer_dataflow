#pragma once

#include "hls_stream.h"
#include "Linear.h"
#include "ScaledDotAtt.h"
#include "Synth_utils.h"

template<typename iT, typename wT, typename tmp1T, typename tmp2T, typename rT, int sequence_length, int token_length, int head_token_length>
void att_head(
	hls::stream<iT> query[token_length],
	hls::stream<iT> key[token_length],
	hls::stream<iT> value[token_length],
	hls::stream<wT> weights[NUM_LINEAR_LAYERS][token_length],
	hls::stream<rT> result[head_token_length]
)
{
	hls::stream<wT, head_token_length> q_weights[token_length]{};
	hls::stream<wT, head_token_length> k_weights[token_length]{};
	hls::stream<wT, head_token_length> v_weights[token_length]{};
	hls::stream<tmp1T, sequence_length> Q[head_token_length]{};
	hls::stream<tmp1T, sequence_length> K[head_token_length]{};
	hls::stream<tmp1T, sequence_length> V[head_token_length]{};

	#pragma HLS DATAFLOW
	split3<wT, head_token_length, token_length>(weights, q_weights, k_weights, v_weights);
	matmul_transpose<iT, wT, tmp1T, sequence_length, token_length, head_token_length>(query, q_weights, Q);
	matmul_transpose<iT, wT, tmp1T, sequence_length, token_length, head_token_length>(key, k_weights, K);
	matmul_transpose<iT, wT, tmp1T, sequence_length, token_length, head_token_length>(value, v_weights, V);
	scaledotatt<tmp1T, tmp2T, rT, sequence_length, head_token_length>(Q, K, V, result);
}
