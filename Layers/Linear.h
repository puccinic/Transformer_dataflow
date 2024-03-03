#pragma once

#include "hls_stream.h"
#include "hls_vector.h"
#include "MatMul.h"

template<typename T, int rows, int cols>
void bias_add
(
	hls::stream<hls::vector<T, cols>> &input,
	hls::stream<hls::vector<T, cols>> &biases,
	hls::stream<hls::vector<T, cols>> &result
)
{
	hls::vector<T, cols> in;
	hls::vector<T, cols> b;
	hls::vector<T, cols> res;

	biases.read(b);

loop_bias_add:
	for (int i = 0; i < rows; i++)
	{
		input.read(in);
		res = in + b;
		result.write(res);
	}
}

template<typename T, int rows, int hidden, int cols>
void linear
(
	hls::stream<hls::vector<T, hidden>> &input,
	hls::stream<hls::vector<T, hidden>> &weights,
	hls::stream<hls::vector<T, cols>>   &biases,
	hls::stream<hls::vector<T, cols>>   &result
)
{
	hls::stream<hls::vector<T, cols>> linear_tmp;

	#pragma HLS DATAFLOW
	matmul_transpose<T, rows, hidden, cols>(input, weights, linear_tmp);
	bias_add<T, rows, cols>(linear_tmp, biases, result);
}
