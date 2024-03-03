#pragma once

#include "hls_stream.h"
#include "hls_vector.h"
#include "Linear.h"
#include "Activations.h"

template<typename T, int rows, int hidden, int cols>
void ff
(
	hls::stream<hls::vector<T, cols>> &input,
	hls::stream<hls::vector<T, cols>> &weights1,
	hls::stream<hls::vector<T, hidden>> &biases1,
	hls::stream<hls::vector<T, hidden>> &weights2,
	hls::stream<hls::vector<T, cols>> &biases2,
	hls::stream<hls::vector<T, cols>> &result
)
{
	hls::stream<hls::vector<T, hidden>> ff_tmp1("ff_tmp1");
	hls::stream<hls::vector<T, hidden>> ff_tmp2("ff_tmp2");

	#pragma HLS DATAFLOW
	linear<T, rows, cols, hidden>(input, weights1, biases1, ff_tmp1);
	activation<T, rows, hidden>(ff_tmp1, ff_tmp2);
	linear<T, rows, hidden, cols>(ff_tmp2, weights2, biases2, result);
}
