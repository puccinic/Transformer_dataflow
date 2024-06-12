#pragma once

#include "hls_stream.h"
#include "MatMul.h"

//TODO: BIAS ADD
template<typename T, int rows, int cols>
void bias_add(
	hls::stream<T> input[cols],
	hls::stream<T> biases[cols],
	hls::stream<T> result[cols]
)
{
	T in;
	T b[cols]{};
	T res;

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

template<typename T, int rows, int hidden, int cols>
void linear(
	hls::stream<T> input[hidden],
	hls::stream<T> weights[hidden],
	hls::stream<T> biases[cols],
	hls::stream<T> result[cols]
)
{
	hls::stream<T, rows> linear_tmp[cols]{};

	#pragma HLS DATAFLOW
	matmul_transpose<T, rows, hidden, cols>(input, weights, linear_tmp);
	bias_add<T, rows, cols>(linear_tmp, biases, result);
}
