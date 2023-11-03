#pragma once
#include <cmath>
template<typename T, size_t size>

struct Norm {
	double epsilon = 1e-5; 
	double gamma = 1.0; 
	double beta = 0.0;
	void init(double epsilon, double gamma, double beta) {
		this->epsilon = epsilon;
		this->gamma = gamma;
		this->beta = beta;
	}
	void operator()(T input[size], T result[size]) {
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
};

template<typename T, size_t channels, size_t size>
struct LayerNorm {
	Norm<T, size> norm_channels[channels];
	
	void init(double epsilon[channels], double gamma[channels], double beta[channels]) {
		for (size_t i = 0; i < channels; i++) {
			norm_channels[i].init(epsilon[i], gamma[i], beta[i]);
		}
	}

	void operator()(T input[channels][size], T result[channels][size]) {
		for (size_t i = 0; i < channels; i++) {
			norm_channels[i](input[i], result[i]);
		}
	}
};