#pragma once
#include <cmath>

template<typename T, size_t channels, size_t size>
void layer_norm(
	T input[channels][size], 
	double epsilon, 
	double gamma[size], 
	double beta[size], 
	T result[channels][size]
) {
layer_norm_outer_loop:
	for (size_t i = 0; i < channels; i++) {
		double sum = 0.0;
	layer_norm_avg_loop:
		for (size_t j = 0; j < size; j++) {
			sum += (double) input[i][j];
		}
		double mean = sum / size;
		double variance = 0.0;
	layer_norm_variance_loop:
		for (size_t j = 0; j < size; j++) {
			double tmp = (((double) input[i][j]) - mean);
			variance += tmp * tmp;
		}
		variance = variance / size;
	layer_norm_result_loop:
		for (size_t j = 0; j < size; j++) {
			result[i][j] = (((input[i][j] - mean) * gamma[j]) / std::sqrt(variance + epsilon)) + beta[j];
		}
	}
}
