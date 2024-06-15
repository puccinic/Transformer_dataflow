#pragma once

#include "hls_stream.h"
#include "AttHead.h"
#include "Concat.h"
#include "Linear.h"

template<typename T, int num_heads, int sequence_length, int token_length, int head_token_length>
void attention_loop(
	hls::stream<T> query[token_length],
	hls::stream<T> key[token_length],
	hls::stream<T> values[token_length],
	hls::stream<T> head_weights[num_heads][NUM_LINEAR_LAYERS][token_length],
	hls::stream<T> result[num_heads][head_token_length]
)
{
	hls::stream<T> query_n[num_heads][token_length]{};
	#pragma HLS STREAM variable=query_n depth=sequence_length
	hls::stream<T> key_n[num_heads][token_length]{};
	#pragma HLS STREAM variable=key_n depth=sequence_length
	hls::stream<T> values_n[num_heads][token_length]{};
	#pragma HLS STREAM variable=values_n depth=sequence_length

	replicate<T, sequence_length, token_length, num_heads>(key, key_n);
	replicate<T, sequence_length, token_length, num_heads>(query, query_n);
	replicate<T, sequence_length, token_length, num_heads>(values, values_n);

	multi_head_att_loop:
	for (int i = 0; i < num_heads; i++)
	{
		att_head<T, sequence_length, token_length, head_token_length>
		(
			query_n[i],
			key_n[i],
			values_n[i],
			head_weights[i],
			result[i]
		);
	}
}

template<typename T, int num_heads, int sequence_length, int token_length, int head_token_length>
void multi_head_att(
	hls::stream<T> query[token_length],
	hls::stream<T> key[token_length],
	hls::stream<T> values[token_length],
	hls::stream<T> head_weights[num_heads][NUM_LINEAR_LAYERS][token_length],
	hls::stream<T> linear_weights[head_token_length*num_heads],
	hls::stream<T> linear_bias[token_length],
	hls::stream<T> result[token_length]
)
{
	hls::stream<T> multihead_tmp1[num_heads][head_token_length]{};
	#pragma HLS STREAM variable=multihead_tmp1 depth=sequence_length
	hls::stream<T> multihead_tmp2[head_token_length*num_heads]{};
	#pragma HLS STREAM variable=multihead_tmp2 depth=sequence_length

	#pragma HLS DATAFLOW
	attention_loop<T, num_heads, sequence_length, token_length, head_token_length>(
		query,
		key,
		values,
		head_weights,
		multihead_tmp1
	);
	concat_cols<T, sequence_length, head_token_length, num_heads>(
		multihead_tmp1,
		multihead_tmp2
	);
	linear<T, sequence_length, head_token_length*num_heads, token_length>(
		multihead_tmp2,
		linear_weights,
		linear_bias,
		result
	);
}
