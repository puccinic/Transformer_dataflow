#pragma once
template<typename T, int rows, int hidden, int cols>
void matmul(T A[rows][hidden], T B[hidden][cols], T result[rows][cols]) {
	//#pragma HLS ARRAY_PARTITION variable = A dim = 2 complete
	//#pragma HLS ARRAY_PARTITION variable = B dim = 1 complete
matmul_row_loop:
	for (int i = 0; i < rows; i++) {
		//#pragma HLS PIPELINE
	matmul_col_loop:
		for (int j = 0; j < cols; j++) {
			//#pragma HLS PIPELINE
			result[i][j] = 0;
		matmul_result_loop:
			for (int k = 0; k < hidden; k++) {
				//#pragma HLS UNROLL
				result[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

template<typename T, int rows, int hidden, int cols>
void transpose_matmul(T A[rows][hidden], T B[cols][hidden], T result[rows][cols]) {
	//#pragma HLS ARRAY_PARTITION variable = A dim = 2 complete
	//#pragma HLS ARRAY_PARTITION variable = B dim = 2 complete
matmul_transpose_row_loop:
	for (int i = 0; i < rows; i++) {
		//#pragma HLS PIPELINE
	matmul_transpose_col_loop:
		for (int j = 0; j < cols; j++) {
			#pragma HLS PIPELINE
			result[i][j] = 0;
		matmul_transpose_result_loop:
			for (int k = 0; k < hidden; k++) {
				//#pragma HLS UNROLL
				result[i][j] += (T) (A[i][k] * B[j][k]);
			}
		}
	}
}

template<typename T, int rows, int hidden, int cols>
void matmul_transpose_scale(T A[rows][hidden], T B[cols][hidden], T scale_factor, T result[rows][cols]) {
	//#pragma HLS ARRAY_PARTITION variable = A dim = 2 complete
	//#pragma HLS ARRAY_PARTITION variable = B dim = 2 complete
matmul_transpose_scale_row_loop:
	for (int i = 0; i < rows; i++) {
		//#pragma HLS PIPELINE
	matmul_transpose_scale_col_loop:
		for (int j = 0; j < cols; j++) {
			//#pragma HLS PIPELINE
			T sum = 0;
		matmul_transpose_scale_result_loop:
			for (int k = 0; k < hidden; k++) {
				//#pragma HLS UNROLL
				sum += (T) (A[i][k] * B[j][k]);
			}
			result[i][j] = sum / scale_factor;
		}
	}
}