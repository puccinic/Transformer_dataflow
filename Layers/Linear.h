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
	iT in[rows][cols]{};
	bT b[cols]{};
	rT res[rows][cols];


	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			#pragma HLS UNROLL
			input.read(in[i][j]);
		}
	}

bias_add_bias_load_loop:
	for (int i = 0; i < cols; i++)
	{
		#pragma HLS UNROLL
		biases.read(b[i]);
	}

bias_add_bias_compute_loop1:
	for (int i = 0; i < rows; i++)
	{
		#pragma HLS UNROLL
	bias_add_bias_compute_loop2:
		for (int j = 0; j < cols; j++)
		{
			#pragma HLS UNROLL
			res[i][j] = in[i][j] + b[j];
		}
	}

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			#pragma HLS UNROLL
			result.write(res[i][j]);
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
	hls::stream<tmpT, rows*cols> linear_tmp{};

	#pragma HLS DATAFLOW
	matmul_transpose<iT, wT, tmpT, rows, hidden, cols>(input, weights, linear_tmp);
	bias_add<tmpT, bT, rT, rows, cols>(linear_tmp, biases, result);
}
