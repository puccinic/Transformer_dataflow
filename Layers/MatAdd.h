#pragma once

#include "hls_stream.h"
#include "hls_vector.h"
#include <ap_fixed.h>

template<
	int bitWidthA,
	int intWidthA,
	int bitWidthB,
	int intWidthB,
	int bitWidthR,
	int intWidthR,
	int rows,
	int cols
>
void matadd(
	hls::stream<hls::vector<ap_fixed<bitWidthA, intWidthA>, cols>>& A,
	hls::stream<hls::vector<ap_fixed<bitWidthB, intWidthB>, cols>>& B,
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, cols>>& result
)
{
	hls::vector<ap_fixed<bitWidthA, intWidthA>, cols> a;
	hls::vector<ap_fixed<bitWidthB, intWidthB>, cols> b;
matadd_loop:
	for (int i = 0; i < rows; i++)
	{
		A.read(a);
	 	B.read(b);
	 	result.write(a + b);
	}
}