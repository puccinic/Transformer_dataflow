#pragma once

#include "hls_stream.h"

template<typename aT, typename bT, typename rT, int rows, int cols>
void matadd(
	hls::stream<aT>& A,
	hls::stream<bT>& B,
	hls::stream<rT>& result
)
{
	static aT a[rows][cols]{};
	static bT b[rows][cols]{};
	static rT rst[rows][cols]{};

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			A.read(a[i][j]);
			B.read(b[i][j]);
		}
	}

matadd_loop1:
	for (int i = 0; i < rows; i++)
	{
		#pragma HLS UNROLL
	matadd_loop2:
		for (int j = 0; j < cols; j++)
		{
			#pragma HLS UNROLL
			rst[i][j] = a[i][j] + b[i][j];
		}
	}

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
	 		result.write(rst[i][j]);
		}
	}
}
