#pragma once

#include "hls_stream.h"
#include "hls_vector.h"
#include "Linear.h"
#include "Activations.h"

template<
	int bitWidthI,
	int intWidthI,
	int bitWidthW1,
	int intWidthW1,
	int bitWidthB1,
	int intWidthB1,
	int bitWidthW2,
	int intWidthW2,
	int bitWidthB2,
	int intWidthB2,
	int bitWidthR,
	int intWidthR,
	int rows,
	int hidden,
	int cols
>
void ff(
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, cols>> &input,
	hls::stream<hls::vector<ap_fixed<bitWidthW1, intWidthW1>, cols>> &weights1,
	hls::stream<hls::vector<ap_fixed<bitWidthB1, intWidthB1>, hidden>> &biases1,
	hls::stream<hls::vector<ap_fixed<bitWidthW2, intWidthW2>, hidden>> &weights2,
	hls::stream<hls::vector<ap_fixed<bitWidthB2, intWidthB2>, cols>> &biases2,
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, cols>> &result
)
{
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, hidden>, rows> ff_tmp1("ff_tmp1");
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, hidden>, rows> ff_tmp2("ff_tmp2");

	#pragma HLS DATAFLOW
	linear<bitWidthI, intWidthI, bitWidthW1, intWidthW1, bitWidthB1, intWidthB1, bitWidthR, intWidthR, rows, cols, hidden>(
		input,
		weights1,
		biases1,
		ff_tmp1
	);
	activation<ap_fixed<bitWidthR, intWidthR>, rows, hidden>(ff_tmp1, ff_tmp2);
	linear<bitWidthR, intWidthR, bitWidthW2, intWidthW2, bitWidthB2, intWidthB2, bitWidthR, intWidthR, rows, hidden, cols>(
		ff_tmp2,
		weights2,
		biases2,
		result
	);
}
