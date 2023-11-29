
#pragma once 
template<typename T, size_t rows, size_t cols>
void matadd(T A[rows][cols], T B[rows][cols], T result[rows][cols]) {
matadd_outer_loop:
	for (size_t i = 0; i < rows; i++) {
	matadd_inner_loop:
		for (size_t j = 0; j < cols; j++) {
			result[i][j] = A[i][j] + B[i][j];
		}
	}
}