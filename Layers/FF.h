#pragma once

#include "hls_stream.h"
#include "Linear.h"
#include "Activations.h"

template<typename T, int rows, int hidden, int cols>
void ff(
	hls::stream<T> input[cols],
	hls::stream<T> weights1[cols],
	hls::stream<T> biases1[hidden],
	hls::stream<T> weights2[hidden],
	hls::stream<T> biases2[cols],
	hls::stream<T> result[cols]
)
{
	hls::stream<T, rows> ff_tmp1[hidden]{};
	hls::stream<T, rows> ff_tmp2[hidden]{};

	#pragma HLS DATAFLOW
	linear<T, rows, cols, hidden>(input, weights1, biases1, ff_tmp1);
	activation<T, rows, hidden>(ff_tmp1, ff_tmp2);
	linear<T, rows, hidden, cols>(ff_tmp2, weights2, biases2, result);
}
