#pragma once
#include <cmath>

template<typename T, int channels, int size>
void layer_norm(
	T input[channels][size], 
	T epsilon, 
	T gamma[size], 
	T beta[size], 
	T result[channels][size]
) {
layer_norm_outer_loop:
	for (int i = 0; i < channels; i++) {
		T sum = 0.0;
	layer_norm_avg_loop:
		for (int j = 0; j < size; j++) {
			sum += (double) input[i][j];
		}
		T mean = sum / size;
		T variance = 0.0;
	layer_norm_variance_loop:
		for (int j = 0; j < size; j++) {
			T tmp = (((double) input[i][j]) - mean);
			variance += tmp * tmp;
		}
		variance = variance / size;
	layer_norm_result_loop:
		for (int j = 0; j < size; j++) {
			result[i][j] = (((input[i][j] - mean) * gamma[j]) / std::sqrt(variance + epsilon)) + beta[j];
		}
	}
}
