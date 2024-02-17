#pragma once

#include "hls_stream.h"
#include "hls_vector.h"
#include "AttHead.h"
#include "Concat.h"
#include "Linear.h"

template<typename T, int num_heads, int sequence_length, int token_length, int head_token_length>
void multi_head_att
(
	hls::stream<hls::vector<T, token_length>> &query,
	hls::stream<hls::vector<T, token_length>> &key,
	hls::stream<hls::vector<T, token_length>> &values,
	hls::stream<hls::vector<T, sequence_length>> &input_mask,
	hls::stream<hls::vector<T, token_length>> head_weights[num_heads][NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<T, head_token_length>> head_biases[num_heads][NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<T, token_length>> &linear_weights,
	hls::stream<hls::vector<T, token_length>> &linear_bias,
	hls::stream<hls::vector<T, token_length>> &result
)
{
	hls::stream<hls::vector<T, head_token_length>> tmp1[num_heads];
	hls::stream<hls::vector<T, head_token_length>> tmp2;

multi_head_att_loop:
	for (int i = 0; i < num_heads; i++)
	{
		att_head<T, sequence_length, token_length, head_token_length>
		(
			query,
			key,
			values,
			input_mask,
			head_weights[i],
			head_biases[i],
			tmp1[i]
		);
	}
	concat_cols<T, sequence_length, head_token_length, num_heads>(tmp1, tmp2);
	linear<T, sequence_length, token_length, token_length>(tmp2, linear_weights, linear_bias, result);
}
