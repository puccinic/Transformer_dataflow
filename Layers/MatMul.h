#pragma once
template<typename T, int rows, int hidden, int cols>
void matmul(T A[rows][hidden], T B[hidden][cols], T result[rows][cols]) {
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