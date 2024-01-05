#pragma once 
template<typename T, int size>
void softmax(T input[size], T result[size]) {
	T sum = 0;
	T tmp[size];
	for (int i = 0; i < size; i++) {
		tmp[i] = (T) hls::exp((double) input[i]);
		sum += (T) hls::exp((double) input[i]);
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
		#pragma HLS PIPELINE rewind
		tmp[i] = mask[i] ? (T) hls::exp((double) input[i]) : (T) 0;
		sum += tmp[i];
	}
softmax_result_loop:
	for (int i = 0; i < size; i++) {
		#pragma HLS PIPELINE rewind
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
	#pragma HLS ARRAY_PARTITION variable = A dim = 2 complete
	#pragma HLS ARRAY_PARTITION variable = B dim = 2 complete
	#pragma HLS ARRAY_PARTITION variable = input_mask dim = 2 complete
matmul_transpose_scale_row_loop:
	for (int i = 0; i < rows; i++) {
		T tmp[cols];
	matmul_transpose_scale_col_loop:
		for (int j = 0; j < cols; j++) {
			T sum = 0;
		matmul_transpose_scale_result_loop:
			for (int k = 0; k < hidden; k++) {
				#pragma HLS PIPELINE rewind
				sum += (A[i][k] * B[j][k]);
			}
			tmp[j] = sum / scale_factor;
		}
		masked_sofmax<T,cols>(tmp, input_mask[i], result[i]);
	}
}