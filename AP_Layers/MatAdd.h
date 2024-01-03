#pragma once 
#include <ap_fixed.h>
template<int width, int i_width, int rows, int cols>
void matadd(
	ap_fixed<width, i_width> A[rows][cols], 
	ap_fixed<width, i_width> B[rows][cols], 
	ap_fixed<width + 1, i_width> result[rows][cols]
) {
matadd_outer_loop:
	for (int i = 0; i < rows; i++) {
	matadd_inner_loop:
		for (int j = 0; j < cols; j++) {
			result[i][j] = A[i][j] + B[i][j];
		}
	}
}