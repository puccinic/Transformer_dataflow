#pragma once

#include "hls_stream.h"

template<typename aT, typename bT, typename rT, int rows, int cols>
void matadd(
	hls::stream<aT>& A,
	hls::stream<bT>& B,
	hls::stream<rT>& result
)
{
	aT a;
	bT b;
	rT rst;
matadd_loop1:
	for (int i = 0; i < rows; i++)
	{
		#pragma HLS UNROLL factor=rows/64
	matadd_loop2:
		for (int j = 0; j < cols; j++)
		{
			#pragma HLS UNROLL factor=cols/64
			A.read(a);
	 		B.read(b);
			rst = a + b;
	 		result.write(rst);
		}
	}
}
