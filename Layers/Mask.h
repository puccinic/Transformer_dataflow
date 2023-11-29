#pragma once
#include <iostream>
#include <vector>

// Apply a mask to a tensor
template<typename T, size_t rows, size_t cols>
void mask(T input[rows][cols], T mask[rows][cols], T result[rows][cols]) {
mask_outer_loop:
	for (size_t i = 0; i < rows; i++) {
	mask_inner_loop:
		for (size_t j = 0; j < cols; j++) {
			result[i][j] = input[i][j] * mask[i][j];
		}
	}
}
