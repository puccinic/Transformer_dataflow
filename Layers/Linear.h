#pragma once

#include "hls_stream.h"
#include "MatMul.h"

template<typename iT, typename bT, typename rT, int rows, int cols>
void bias_add(
	hls::stream<iT>& input,
	hls::stream<bT>& biases,
	hls::stream<rT>& result
)
{
	iT in;
	bT b[cols]{};
	rT res;

bias_add_bias_load_loop:
	for (int i = 0; i < cols; i++)
	{
		#pragma HLS UNROLL factor=cols/64
		biases.read(b[i]);
	}

bias_add_bias_compute_loop1:
	for (int i = 0; i < rows; i++)
	{
		#pragma HLS UNROLL factor=rows/64
	bias_add_bias_compute_loop2:
		for (int j = 0; j < cols; j++)
		{
			#pragma HLS UNROLL factor=cols/64
			input.read(in);
			res = in + b[j];
			result.write(res);
		}
	}
}

template<typename iT, typename wT, typename bT, typename tmpT, typename rT, int rows, int hidden, int cols>
void linear(
	hls::stream<iT>& input,
	hls::stream<wT>& weights,
	hls::stream<bT>& biases,
	hls::stream<rT>& result
)
{
	hls::stream<tmpT, rows> linear_tmp{};

	#pragma HLS DATAFLOW
	matmul_transpose<iT, wT, tmpT, rows, hidden, cols>(input, weights, linear_tmp);
	bias_add<tmpT, bT, rT, rows, cols>(linear_tmp, biases, result);
}
