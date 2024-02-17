#pragma once

#include "hls_stream.h"
#include "hls_vector.h"

template<typename T, int rows, int cols>
void matadd(
	hls::stream<hls::vector<T, cols>>& A,
	hls::stream<hls::vector<T, cols>>& B,
	hls::stream<hls::vector<T, cols>>& result
)
{
matadd_loop:
	hls::vector<T, cols> a;
	hls::vector<T, cols> b;
	for (int i = 0; i < rows; i++)
	{
		A.read(a);
	 	B.read(b);
	 	result.write(a + b);
	}
}