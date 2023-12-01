
#pragma once 
template<typename T, int rows, int cols>
void matadd(T A[rows][cols], T B[rows][cols], T result[rows][cols]) {
matadd_outer_loop:
	for (int i = 0; i < rows; i++) {
	matadd_inner_loop:
		for (int j = 0; j < cols; j++) {
			result[i][j] = A[i][j] + B[i][j];
		}
	}
}