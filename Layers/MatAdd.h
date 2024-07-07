#pragma once

#include "hls_stream.h"

template<typename aT, typename bT, typename rT, int rows, int cols>
void matadd(
	hls::stream<aT> A[cols],
	hls::stream<bT> B[cols],
	hls::stream<rT> result[cols]
)
{
	aT a;
	bT b;
	rT rst;
matadd_loop1:
	for (int i = 0; i < rows; i++)
	{
	matadd_loop2:
		for (int j = 0; j < cols; j++)
		{
			A[j].read(a);
	 		B[j].read(b);
			rst = a + b;
	 		result[j].write(rst);
		}
	}
}