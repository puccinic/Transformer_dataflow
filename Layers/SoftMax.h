#pragma once 
#include <cmath>

template<typename T, int size>
void softmax(T input[size], T result[size]) {
	T sum = 0;
	T tmp[size];
	for (int i = 0; i < size; i++) {
		tmp[i] = std::exp(input[i]);
		sum += std::exp(input[i]);
	}

	for (int i = 0; i < size; i++) {
		result[i] = (T)((tmp[i] / sum));
	}
}

template<typename T, int size>
void masked_sofmax(T input[size], T mask[size], T result[size]) {
	T sum = 0;
	T tmp[size];
softmax_sum_loop:
	for (int i = 0; i < size; i++) {
		tmp[i] = mask[i] ? std::exp(input[i]) : 0;
		sum += tmp[i];
	}
softmax_result_loop:
	for (int i = 0; i < size; i++) {
		result[i] = (T)((tmp[i] / sum));
	}
}


template<typename T, int rows, int hidden, int cols>
void matmul_scale_masked_softmax(
	T A[rows][hidden], 
	T B[cols][hidden], 
	T scale_factor, 
	T input_mask[rows][cols],
	T result[rows][cols]) {
matmul_transpose_scale_row_loop:
	for (int i = 0; i < rows; i++) {
		T tmp[cols];
	matmul_transpose_scale_col_loop:
		for (int j = 0; j < cols; j++) {
			T sum = 0;
		matmul_transpose_scale_result_loop:
			for (int k = 0; k < hidden; k++) {
				sum += A[i][k] * B[j][k];
			}
			tmp[j] = sum / scale_factor;
		}
		masked_sofmax<T,cols>(tmp, input_mask[i], result[i]);
	}
}