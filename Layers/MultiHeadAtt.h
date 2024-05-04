#pragma once

#include "hls_stream.h"
#include "hls_vector.h"
#include "AttHead.h"
#include "Concat.h"
#include "Linear.h"

template<
	int bitWidthI,
	int intWidthI,
	int bitWidthW,
	int intWidthW,
	int bitWidthB,
	int intWidthB,
	int bitWidthR,
	int intWidthR,
	int num_heads,
	int sequence_length,
	int token_length,
	int head_token_length
>
void attention_loop(
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, token_length>> &query,
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, token_length>> &key,
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, token_length>> &values,
	hls::stream<hls::vector<int, sequence_length>> &input_mask,
	hls::stream<hls::vector<ap_fixed<bitWidthW, intWidthW>, token_length>> head_weights[num_heads][NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<ap_fixed<bitWidthB, intWidthB>, head_token_length>> head_biases[num_heads][NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, head_token_length>> result[num_heads]
)
{
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, token_length>, sequence_length> query_n[num_heads];
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, token_length>, sequence_length> key_n[num_heads];
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, token_length>, sequence_length> values_n[num_heads];
	hls::stream<hls::vector<int, sequence_length>, sequence_length> mask_n[num_heads];

	replicate<ap_fixed<bitWidthI, intWidthI>, sequence_length, token_length, num_heads>(key, key_n);
	replicate<ap_fixed<bitWidthI, intWidthI>, sequence_length, token_length, num_heads>(query, query_n);
	replicate<ap_fixed<bitWidthI, intWidthI>, sequence_length, token_length, num_heads>(values, values_n);
	replicate<int, sequence_length, sequence_length, num_heads>(input_mask, mask_n);

	multi_head_att_loop:
	for (int i = 0; i < num_heads; i++)
	{
		att_head<bitWidthI, intWidthI, bitWidthW, intWidthW, bitWidthB, intWidthB, bitWidthR, intWidthR, sequence_length, token_length, head_token_length>
		(
			query_n[i],
			key_n[i],
			values_n[i],
			mask_n[i],
			head_weights[i],
			head_biases[i],
			result[i]
		);
	}
}

template<
	int bitWidthI,
	int intWidthI,
	int bitWidthW1,
	int intWidthW1,
	int bitWidthB1,
	int intWidthB1,
	int bitWidthW2,
	int intWidthW2,
	int bitWidthB2,
	int intWidthB2,
	int bitWidthR,
	int intWidthR,
	int num_heads,
	int sequence_length,
	int token_length,
	int head_token_length
>
void multi_head_att(
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, token_length>> &query,
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, token_length>> &key,
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, token_length>> &values,
	hls::stream<hls::vector<int, sequence_length>> &input_mask,
	hls::stream<hls::vector<ap_fixed<bitWidthW1, intWidthW1>, token_length>> head_weights[num_heads][NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<ap_fixed<bitWidthB1, intWidthB1>, head_token_length>> head_biases[num_heads][NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<ap_fixed<bitWidthW2, intWidthW2>, token_length>> &linear_weights,
	hls::stream<hls::vector<ap_fixed<bitWidthB2, intWidthB2>, token_length>> &linear_bias,
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, token_length>> &result
)
{
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, head_token_length>, sequence_length> multihead_tmp1[num_heads];
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, token_length>, sequence_length> multihead_tmp2("multihead_tmp2");

	#pragma HLS DATAFLOW
	attention_loop<bitWidthI, intWidthI, bitWidthW1, intWidthW1, bitWidthB1, intWidthB1, bitWidthR, intWidthR, num_heads, sequence_length, token_length, head_token_length>(
		query,
		key,
		values,
		input_mask,
		head_weights,
		head_biases,
		multihead_tmp1
	);
	concat_cols<ap_fixed<bitWidthR, intWidthR>, sequence_length, head_token_length, num_heads>(
		multihead_tmp1,
		multihead_tmp2
	);
	linear<bitWidthR, intWidthR, bitWidthW2, intWidthW2, bitWidthB2, intWidthB2, bitWidthR, intWidthR, sequence_length, token_length, token_length>(
		multihead_tmp2,
		linear_weights,
		linear_bias,
		result
	);
}
