#pragma once
#include <ap_fixed.h>
#ifndef OP
#define OP /
#endif // !OP

template<int width, int i_width, int rows, int cols>
void scale(
	ap_fixed<width, i_width> A[rows][cols], 
	ap_fixed<width, i_width> scale_factor,
	ap_fixed<width*2, i_width*2> result[rows][cols]
) {
scale_outer_loop:
	for (int i = 0; i < rows; i++) {
	scale_inner_loop:
		for (int j = 0; j < cols; j++) {
#pragma HLS UNROLL
			result[i][j] = A[i][j] OP scale_factor;
		}
	}
}
