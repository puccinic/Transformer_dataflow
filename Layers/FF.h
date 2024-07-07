#pragma once

#include "hls_stream.h"
#include "Linear.h"
#include "Activations.h"

template<typename iT, typename w1T, typename b1T, typename w2T, typename b2T, typename tmpl1T, typename tmpT, typename tmpl2T, typename rT, int rows, int hidden, int cols>
void ff(
	hls::stream<iT> input[cols],
	hls::stream<w1T> weights1[cols],
	hls::stream<b1T> biases1[hidden],
	hls::stream<w2T> weights2[hidden],
	hls::stream<b2T> biases2[cols],
	hls::stream<rT> result[cols]
)
{
	hls::stream<tmpT, rows> ff_tmp1[hidden]{};
	hls::stream<tmpT, rows> ff_tmp2[hidden]{};

	#pragma HLS DATAFLOW
	linear<iT, w1T, b1T, tmpl1T, tmpT, rows, cols, hidden>(input, weights1, biases1, ff_tmp1);
	activation<tmpT, rows, hidden>(ff_tmp1, ff_tmp2);
	linear<tmpT, w2T, b2T, tmpl2T, rT, rows, hidden, cols>(ff_tmp2, weights2, biases2, result);
}
