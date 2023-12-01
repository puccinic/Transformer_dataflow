#pragma once
template<typename T, int rows, int cols>

#ifndef OP
#define OP /
#endif // !OP

void scale(T A[rows][cols], T result[rows][cols], T scale_factor) {
scale_outer_loop:
	for (int i = 0; i < rows; i++) {
	scale_inner_loop:
		for (int j = 0; j < cols; j++) {
#pragma HLS UNROLL
			result[i][j] = A[i][j] OP scale_factor;
		}
	}
}
