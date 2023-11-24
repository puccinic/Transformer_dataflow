#pragma once
#include <cmath>

template<typename T, size_t channels, size_t size>
struct LayerNorm {
	double epsilon = 1e-5;
	double* gamma;
	double* beta;
	void init(double epsilon, double gamma[size], double beta[size]) {
		this->epsilon = epsilon;
		this->gamma = gamma;
		this->beta = beta;
	}

	void layer_norm_function(T input[size], T result[size]) {
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
			result[i] = (((input[i] - mean) * gamma[i]) / std::sqrt(variance + epsilon)) + beta[i];
		}
	}

	void operator()(T input[channels][size], T result[channels][size]) {
		for (size_t i = 0; i < channels; i++) {
			layer_norm_function(input[i], result[i]);
		}
	}
};