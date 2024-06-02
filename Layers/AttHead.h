#pragma once

#include "hls_stream.h"
#include "hls_vector.h"
#include "Linear.h"
#include "ScaledDotAtt.h"
#include "Synth_utils.h"

template<typename T, int sequence_length, int token_length, int head_token_length>
void att_head(
	hls::stream<hls::vector<T, token_length>> &query,
	hls::stream<hls::vector<T, token_length>> &key,
	hls::stream<hls::vector<T, token_length>> &value,
	hls::stream<hls::vector<T, token_length>> weights[NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<T, head_token_length>> &result
)
{
	hls::stream<hls::vector<T, token_length>, head_token_length> q_weights("q_weights");
	hls::stream<hls::vector<T, token_length>, head_token_length> k_weights("k_weights");
	hls::stream<hls::vector<T, token_length>, head_token_length> v_weights("v_weights");
	hls::stream<hls::vector<T, head_token_length>, sequence_length> Q("Q");
	hls::stream<hls::vector<T, head_token_length>, sequence_length> K("K");
	hls::stream<hls::vector<T, head_token_length>, sequence_length> V("V");

	#pragma HLS DATAFLOW
	split3<T, head_token_length, token_length>(weights, q_weights, k_weights, v_weights);
	matmul_transpose<T, sequence_length, token_length, head_token_length>(query, q_weights, Q);
	matmul_transpose<T, sequence_length, token_length, head_token_length>(key, k_weights, K);
	matmul_transpose<T, sequence_length, token_length, head_token_length>(value, v_weights, V);
	scaledotatt<T, sequence_length, head_token_length>(Q, K, V, result);
}
