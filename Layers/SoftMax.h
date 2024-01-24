#pragma once


template <typename T, int size>
void get_max(T input[size], T *result) {
	T tmp = 0;
	for (int i = 1; i < size; i++) {
		if (tmp < input[i]) {
      	tmp = input[i];
    	}
  	}
	*result = tmp;
}

template<typename T, int size>
void softmax(T input[size], T result[size]) {
	T max;
	get_max<T, size>(input, &max);
	T sum = 0;
	T tmp[size];
	for (int i = 0; i < size; i++) {
		tmp[i] = (T) hls::exp((double) (input[i] - max));
		sum += tmp[i];
	}

	for (int i = 0; i < size; i++) {
		result[i] = (T)((tmp[i] / sum));
	}
}

template<typename T, int size>
void masked_sofmax(T input[size], T mask[size], T result[size]) {
	T max = 0;
	get_max<T, size>(input, &max);
	T sum = 0;
	T tmp[size];
softmax_sum_loop:
	for (int i = 0; i < size; i++) {
		//#pragma HLS PIPELINE rewind
		tmp[i] = mask[i] ? (T) hls::exp((double) (input[i] - max)) : (T) 0;
		sum += tmp[i];
	}

softmax_result_loop:
	for (int i = 0; i < size; i++) {
		//#pragma HLS PIPELINE rewind
		if(tmp[i] != 0 && sum != 0) {
			result[i] = tmp[i] / sum;
		} else {
			result[i] = 0;
		}
	}
}


template<typename T, int rows, int hidden, int cols>
void matmul_scale_masked_softmax(
	T A[rows][hidden],
	T B[cols][hidden],
	T scale_factor,
	T input_mask[rows][cols],
	T result[rows][cols]) {
	//#pragma HLS ARRAY_PARTITION variable = A dim = 2 complete
	//#pragma HLS ARRAY_PARTITION variable = B dim = 2 complete
	//#pragma HLS ARRAY_PARTITION variable = input_mask dim = 2 complete
matmul_transpose_scale_row_loop:
	for (int i = 0; i < rows; i++) {
		//#pragma HLS PIPELINE rewind
		T tmp[cols];
	matmul_transpose_scale_col_loop:
		for (int j = 0; j < cols; j++) {
			//#pragma HLS PIPELINE rewind
			T sum = 0;
		matmul_transpose_scale_result_loop:
			for (int k = 0; k < hidden; k++) {
				//#pragma HLS UNROLL
				sum += (A[i][k] * B[j][k]);
			}
			tmp[j] = sum / scale_factor;
		}
		masked_sofmax<T,cols>(tmp, input_mask[i], (T*) result[i]);
	}
}