#pragma once

// Apply a mask to a tensor
template<typename T, int rows, int cols>
void mask(T input[rows][cols], T mask[rows][cols], T result[rows][cols]) {
mask_outer_loop:
	for (int i = 0; i < rows; i++) {
	mask_inner_loop:
		for (int j = 0; j < cols; j++) {
			result[i][j] = input[i][j] * mask[i][j];
		}
	}
}
