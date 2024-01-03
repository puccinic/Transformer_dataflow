#pragma once
#include <ap_fixed.h>
template<int width, int i_width, int size> 
void softmax(
	ap_fixed<width, i_width> input[size], 
	ap_fixed<width, i_width> result[size]
) {
	ap_fixed<width + size, i_width> sum = 0;
	ap_fixed<width, i_width> tmp[size];
	for (int i = 0; i < size; i++) {
		tmp[i] = hls::exp((double) input[i]);
		sum += tmp[i];
	}

	for (int i = 0; i < size; i++) {
		result[i] = ((tmp[i] / sum));
	}
}

template<int width, int i_width, int size> 
void masked_sofmax(
	ap_fixed<width, i_width> input[size], 
	bool mask[size],
	ap_fixed<width, i_width> result[size]
) {
	ap_fixed<width + size, i_width> sum = 0;
	ap_fixed<width, i_width> tmp[size];
softmax_sum_loop:
	for (int i = 0; i < size; i++) {
		tmp[i] = mask[i] ? hls::exp((double) input[i]) : 0;
		sum += tmp[i];
	}
softmax_result_loop:
	for (int i = 0; i < size; i++) {
		result[i] = ((tmp[i] / sum));
	}
}

template<int width, int i_width, int rows, int hidden, int cols>
void matmul_scale_masked_softmax(
	ap_fixed<width, i_width> A[rows][hidden], 
	ap_fixed<width, i_width> B[cols][hidden], 
	ap_fixed<width, i_width> scale_factor, 
	bool input_mask[rows][cols],
	ap_fixed<2*width + hidden, 2*i_width> result[rows][cols]
) {
matmul_transpose_scale_row_loop:
	for (int i = 0; i < rows; i++) {
		ap_fixed<2*width + hidden, 2*i_width> tmp[cols];
	matmul_transpose_scale_col_loop:
		for (int j = 0; j < cols; j++) {
			ap_fixed<2*width + hidden, 2*i_width> sum = 0;
		matmul_transpose_scale_result_loop:
			for (int k = 0; k < hidden; k++) {
				sum += (A[i][k] * B[j][k]);
			}
			tmp[j] = sum / scale_factor;
		}
		masked_sofmax<2*width + hidden, 2*i_width, int i_width,cols>(tmp, input_carlosmask[i], result[i]);
	}
}