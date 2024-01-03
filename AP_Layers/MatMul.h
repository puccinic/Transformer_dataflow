#pragma once
#include <ap_fixed.h>
template<int width, int i_width, int rows, int hidden, int cols>
void matmul(
	ap_fixed<width, i_width> A[rows][hidden], 
	ap_fixed<width, i_width> B[hidden][cols], 
	ap_fixed<2*width + hidden, 2*i_width> result[rows][cols]
) {
matmul_row_loop:
	for (int i = 0; i < rows; i++) {
	matmul_col_loop:
		for (int j = 0; j < cols; j++) {
			result[i][j] = 0;
		matmul_result_loop:
			for (int k = 0; k < hidden; k++) {
				result[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

template<int width, int i_width, int rows, int hidden, int cols>
void transpose_matmul(
	ap_fixed<width, i_width> A[rows][hidden], 
	ap_fixed<width, i_width> B[hidden][cols], 
	ap_fixed<2*width + hidden, 2*i_width> result[rows][cols]
) {
matmul_transpose_row_loop:
	for (int i = 0; i < rows; i++) {
	matmul_transpose_col_loop:
		for (int j = 0; j < cols; j++) {
			result[i][j] = 0;
		matmul_transpose_result_loop:
			for (int k = 0; k < hidden; k++) {
				result[i][j] += (T) (A[i][k] * B[j][k]);
			}
		}
	}
}

template<int width, int i_width, int rows, int hidden, int cols>
void matmul_transpose_scale(
	ap_fixed<width, i_width> A[rows][hidden], 
	ap_fixed<width, i_width> B[hidden][cols],
	ap_fixed<width, i_width> scale_factor, 
	ap_fixed<2*width + hidden, 2*i_width> result[rows][cols]
) {
matmul_transpose_scale_row_loop:
	for (int i = 0; i < rows; i++) {
	matmul_transpose_scale_col_loop:
		for (int j = 0; j < cols; j++) {
			T sum = 0;
		matmul_transpose_scale_result_loop:
			for (int k = 0; k < hidden; k++) {
				sum += (T) (A[i][k] * B[j][k]);
			}
			result[i][j] = sum / scale_factor;
		}
	}
}