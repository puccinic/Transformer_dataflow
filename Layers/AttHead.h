#pragma once

#include "hls_stream.h"
#include "hls_vector.h"
#include "Linear.h"
#include "ScaledDotAtt.h"
#include "Synth_utils.h"

template<
	int bitWidthI,
	int intWidthI,
	int bitWidthW,
	int intWidthW,
	int bitWidthB,
	int intWidthB,
	int bitWidthR,
	int intWidthR,
	int sequence_length,
	int token_length,
	int head_token_length
>
void att_head(
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, token_length>> &query,
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, token_length>> &key,
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, token_length>> &value,
	hls::stream<hls::vector<int, sequence_length>> &input_mask,
	hls::stream<hls::vector<ap_fixed<bitWidthW, intWidthW>, token_length>> weights[NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<ap_fixed<bitWidthB, intWidthB>, head_token_length>> biases[NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, head_token_length>> &result
)
{
	hls::stream<hls::vector<ap_fixed<bitWidthW, intWidthW>, token_length>, head_token_length> q_weights("q_weights");
	hls::stream<hls::vector<ap_fixed<bitWidthW, intWidthW>, token_length>, head_token_length> k_weights("k_weights");
	hls::stream<hls::vector<ap_fixed<bitWidthW, intWidthW>, token_length>, head_token_length> v_weights("v_weights");
	hls::stream<hls::vector<ap_fixed<bitWidthB, intWidthB>, head_token_length>, head_token_length> q_biases("q_biases");
	hls::stream<hls::vector<ap_fixed<bitWidthB, intWidthB>, head_token_length>, head_token_length> k_biases("k_biases");
	hls::stream<hls::vector<ap_fixed<bitWidthB, intWidthB>, head_token_length>, head_token_length> v_biases("v_biases");
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, head_token_length>, sequence_length> Q("Q");
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, head_token_length>, sequence_length> K("K");
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, head_token_length>, sequence_length> V("V");

	#pragma HLS DATAFLOW
	split3<ap_fixed<bitWidthW, intWidthW>, head_token_length, token_length>(weights, q_weights, k_weights, v_weights);
	split3<ap_fixed<bitWidthB, intWidthB>, 1, head_token_length>(biases, q_biases, k_biases, v_biases);
	linear<bitWidthI, intWidthI, bitWidthW, intWidthW, bitWidthB, intWidthB, bitWidthR, intWidthR, sequence_length, token_length, head_token_length>(query, q_weights, q_biases, Q);
	linear<bitWidthI, intWidthI, bitWidthW, intWidthW, bitWidthB, intWidthB, bitWidthR, intWidthR, sequence_length, token_length, head_token_length>(key, k_weights, k_biases, K);
	linear<bitWidthI, intWidthI, bitWidthW, intWidthW, bitWidthB, intWidthB, bitWidthR, intWidthR, sequence_length, token_length, head_token_length>(value, v_weights, v_biases, V);
	scaledotatt<bitWidthR, intWidthR, sequence_length, head_token_length>(Q, K, V, input_mask, result);
}
