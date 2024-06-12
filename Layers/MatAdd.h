#pragma once

#include "hls_stream.h"

template<typename T, int rows, int cols>
void matadd(
	hls::stream<T> A[cols],
	hls::stream<T> B[cols],
	hls::stream<T> result[cols]
)
{
	T a;
	T b;
	T rst;
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