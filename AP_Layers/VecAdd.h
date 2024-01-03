#pragma once
#include <ap_fixed.h>
template<int width, int i_width, int size>
void vecadd(
	ap_fixed<width, i_width> A[size], 
	ap_fixed<width, i_width> B[size], 
	ap_fixed<width + 1, i_width> result[size]
) {
vecadd_loop:
	for (int i = 0; i < size; i++) {
#pragma HLS UNROLL
		result[i] = A[i] + B[i];
	}
}
