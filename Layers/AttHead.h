#pragma once

#include "hls_stream.h"
#include "hls_vector.h"
#include "Linear.h"
#include "ScaledDotAtt.h"
#include "Synth_utils.h"

template<typename T, int sequence_length, int token_length, int head_token_length>
void att_head
(
	hls::stream<hls::vector<T, token_length>> &query,
	hls::stream<hls::vector<T, token_length>> &key,
	hls::stream<hls::vector<T, token_length>> &value,
	hls::stream<hls::vector<T, sequence_length>> &input_mask,
	hls::stream<hls::vector<T, token_length>> weights[NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<T, head_token_length>> biases[NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<T, head_token_length>> &result
)
{
	hls::stream<hls::vector<T, token_length>> q_weights;
	hls::stream<hls::vector<T, token_length>> k_weights;
	hls::stream<hls::vector<T, token_length>> v_weights;
	hls::stream<hls::vector<T, head_token_length>> q_biases;
	hls::stream<hls::vector<T, head_token_length>> k_biases;
	hls::stream<hls::vector<T, head_token_length>> v_biases;
	hls::stream<hls::vector<T, head_token_length>> Q;
	hls::stream<hls::vector<T, head_token_length>> K;
	hls::stream<hls::vector<T, head_token_length>> V;

	#pragma HLS DATAFLOW
	split3<T, head_token_length, token_length>(weights, q_weights, k_weights, v_weights);
	split3<T, 1, head_token_length>(biases, q_biases, k_biases, v_biases);
	linear<T, sequence_length, token_length, head_token_length>(query, q_weights, q_biases, Q);
	linear<T, sequence_length, token_length, head_token_length>(key, k_weights, k_biases, K);
	linear<T, sequence_length, token_length, head_token_length>(value, v_weights, v_biases, V);
	scaledotatt<T, sequence_length, head_token_length>(Q, K, V, input_mask, result);
}
