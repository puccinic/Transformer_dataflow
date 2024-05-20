#pragma once

#include "hls_stream.h"
#include "hls_vector.h"
#include "MatMul.h"

template<
	int bitWidthI,
	int intWidthI,
	int bitWidthB,
	int intWidthB,
	int bitWidthR,
	int intWidthR,
	int rows,
	int cols
>
void bias_add(
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, cols>> &input,
	hls::stream<hls::vector<ap_fixed<bitWidthB, intWidthB>, cols>> &biases,
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, cols>> &result
)
{
	hls::vector<ap_fixed<bitWidthI, intWidthI>, cols> in;
	hls::vector<ap_fixed<bitWidthB, intWidthB>, cols> b;
	hls::vector<ap_fixed<bitWidthR, intWidthR>, cols> res;

	biases.read(b);

loop_bias_add:
	for (int i = 0; i < rows; i++)
	{
		input.read(in);
		for (int j = 0; j < cols; j++)
		{
			res[j] = in[j] + b[j];
		}
		result.write(res);
	}
}

template<
	int bitWidthI,
	int intWidthI,
	int bitWidthW,
	int intWidthW,
	int bitWidthB,
	int intWidthB,
	int bitWidthR,
	int intWidthR,
	int rows,
	int hidden,
	int cols
>
void linear(
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, hidden>> &input,
	hls::stream<hls::vector<ap_fixed<bitWidthW, intWidthW>, hidden>> &weights,
	hls::stream<hls::vector<ap_fixed<bitWidthB, intWidthB>, cols>>   &biases,
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, cols>>   &result
)
{
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, cols>, rows> linear_tmp;

	#pragma HLS DATAFLOW
	matmul_transpose<bitWidthI, intWidthI, bitWidthW, intWidthW, bitWidthR, intWidthR, rows, hidden, cols>(
		input,
		weights,
		linear_tmp
	);
	bias_add<bitWidthR, intWidthR, bitWidthB, intWidthB, bitWidthR, intWidthR, rows, cols>(
		linear_tmp,
		biases,
		result
	);
}
