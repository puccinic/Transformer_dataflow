#pragma once
template<typename T, size_t rows, size_t cols>

#ifndef OP
#define OP /
#endif // !OP

void scale(T A[rows][cols], T result[rows][cols], T scale_factor) {
scale_outer_loop:
	for (size_t i = 0; i < rows; i++) {
	scale_inner_loop:
		for (size_t j = 0; j < cols; j++) {
			result[i][j] = A[i][j] OP scale_factor;
		}
	}
}