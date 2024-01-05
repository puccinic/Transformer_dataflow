
#pragma once 
template<typename T, int rows, int cols>
void matadd(T A[rows][cols], T B[rows][cols], T result[rows][cols]) {
	#pragma HLS ARRAY_PARTITION variable = A dim = 2 complete
	#pragma HLS ARRAY_PARTITION variable = B dim = 2 complete
matadd_outer_loop:
	for (int i = 0; i < rows; i++) {
	matadd_inner_loop:
		for (int j = 0; j < cols; j++) {
			#pragma HLS UNROLL
			result[i][j] = A[i][j] + B[i][j];
		}
	}
}