#pragma once
#include <cmath>
template<typename T, size_t size>
void layernorm(T input[size], T result[size],
	double epsilon = 1e-5, double gamma = 1.0, double beta = 0.0) {

	double sum = 0.0;
	for (size_t i = 0; i < size; i++) {
		sum += (double)input[i];
	}
	double mean = sum / size;
	double variance = 0.0;
	for (size_t i = 0; i < size; i++) {
		double tmp = (((double)input[i]) - mean);
		variance += tmp * tmp;
	}
	variance = variance / size;
	for (size_t i = 0; i < size; i++) {
		result[i] = ((input[i] - mean) / std::sqrt(variance + epsilon)) * gamma + beta;
	}
}