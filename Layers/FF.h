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
	#pragma HLS DATAFLOW
	hls::stream<hls::vector<T, hidden>> &tmp1;
	hls::stream<hls::vector<T, hidden>> &tmp2;
	linear<T, rows, cols, hidden>(input, weights1, biases1, tmp1);
	activation<T, rows, hidden>(tmp1, tmp2);
	linear<T, rows, hidden, cols>(tmp2, weights2, biases2, result);
}
