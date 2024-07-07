#pragma once

#include "hls_stream.h"
#include "AttHead.h"
#include "Concat.h"
#include "Linear.h"

template<
	typename iT,
	typename wT,
	typename tmp1T,
	typename tmp2T,
	typename rT,
	int num_heads,
	int sequence_length,
	int token_length,
	int head_token_length
>
void attention_loop(
	hls::stream<iT> query[token_length],
	hls::stream<iT> key[token_length],
	hls::stream<iT> values[token_length],
	hls::stream<wT> head_weights[num_heads][NUM_LINEAR_LAYERS][token_length],
	hls::stream<rT> result[num_heads][head_token_length]
)
{
	hls::stream<iT> query_n[num_heads][token_length]{};
	#pragma HLS STREAM variable=query_n depth=sequence_length
	hls::stream<iT> key_n[num_heads][token_length]{};
	#pragma HLS STREAM variable=key_n depth=sequence_length
	hls::stream<iT> values_n[num_heads][token_length]{};
	#pragma HLS STREAM variable=values_n depth=sequence_length

	replicate<iT, sequence_length, token_length, num_heads>(key, key_n);
	replicate<iT, sequence_length, token_length, num_heads>(query, query_n);
	replicate<iT, sequence_length, token_length, num_heads>(values, values_n);

	multi_head_att_loop:
	for (int i = 0; i < num_heads; i++)
	{
		att_head<iT, wT, tmp1T, tmp2T, rT, sequence_length, token_length, head_token_length>
		(
			query_n[i],
			key_n[i],
			values_n[i],
			head_weights[i],
			result[i]
		);
	}
}

template<
	typename iT,
	typename whT,
	typename wlT,
	typename blT,
	typename tmpatt1T,
	typename tmpatt2T,
	typename tmpT,
	typename tmplT,
	typename rT,
	int num_heads,
	int sequence_length,
	int token_length,
	int head_token_length
>
void multi_head_att(
	hls::stream<iT> query[token_length],
	hls::stream<iT> key[token_length],
	hls::stream<iT> values[token_length],
	hls::stream<whT> head_weights[num_heads][NUM_LINEAR_LAYERS][token_length],
	hls::stream<wlT> linear_weights[head_token_length*num_heads],
	hls::stream<blT> linear_bias[token_length],
	hls::stream<rT> result[token_length]
)
{
	hls::stream<tmpT> multihead_tmp1[num_heads][head_token_length]{};
	#pragma HLS STREAM variable=multihead_tmp1 depth=sequence_length
	hls::stream<tmpT> multihead_tmp2[head_token_length*num_heads]{};
	#pragma HLS STREAM variable=multihead_tmp2 depth=sequence_length

	#pragma HLS DATAFLOW
	attention_loop<iT, whT, tmpatt1T, tmpatt2T, tmpT, num_heads, sequence_length, token_length, head_token_length>(
		query,
		key,
		values,
		head_weights,
		multihead_tmp1
	);
	concat_cols<tmpT, sequence_length, head_token_length, num_heads>(
		multihead_tmp1,
		multihead_tmp2
	);
	linear<tmpT, wlT, blT, tmplT, rT, sequence_length, head_token_length*num_heads, token_length>(
		multihead_tmp2,
		linear_weights,
		linear_bias,
		result
	);
}
