#pragma once

#include "hls_stream.h"
#include "Linear.h"
#include "Activations.h"

template<typename iT, typename w1T, typename w2T, typename b2T, typename tmpT, typename tmpl2T, typename rT, int rows, int hidden, int cols>
void ff(
	hls::stream<iT>& input,
	hls::stream<w1T>& weights1,
	hls::stream<w2T>& weights2,
	hls::stream<b2T>& biases2,
	hls::stream<rT>& result
)
{
	hls::stream<tmpT, rows> ff_tmp1{};
	hls::stream<tmpT, rows> ff_tmp2{};

	#pragma HLS DATAFLOW
	matmul_transpose<iT, w1T, tmpT, rows, cols, hidden>(input, weights1, ff_tmp1);
	activation<tmpT, rows, hidden>(ff_tmp1, ff_tmp2);
	linear<tmpT, w2T, b2T, tmpl2T, rT, rows, hidden, cols>(ff_tmp2, weights2, biases2, result);
}
