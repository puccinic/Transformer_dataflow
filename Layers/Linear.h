#pragma once

#include "hls_stream.h"
#include "MatMul.h"

template<typename iT, typename bT, typename rT, int rows, int cols>
void bias_add(
	hls::stream<iT> input[cols],
	hls::stream<bT> biases[cols],
	hls::stream<rT> result[cols]
)
{
	iT in;
	bT b[cols]{};
	rT res;

bias_add_bias_load_loop:
	for (int i = 0; i < cols; i++)
	{
		biases[i].read(b[i]);
	}

bias_add_bias_compute_loop1:
	for (int i = 0; i < rows; i++)
	{
	bias_add_bias_compute_loop2:
		for (int j = 0; j < cols; j++)
		{
			input[j].read(in);
			res = in + b[j];
			result[j].write(res);
		}
	}
}

template<typename iT, typename wT, typename bT, typename tmpT, typename rT, int rows, int hidden, int cols>
void linear(
	hls::stream<iT> input[hidden],
	hls::stream<wT> weights[hidden],
	hls::stream<bT> biases[cols],
	hls::stream<rT> result[cols]
)
{
	hls::stream<tmpT, rows> linear_tmp[cols]{};

	#pragma HLS DATAFLOW
	matmul_transpose<iT, wT, tmpT, rows, hidden, cols>(input, weights, linear_tmp);
	bias_add<tmpT, bT, rT, rows, cols>(linear_tmp, biases, result);
}
