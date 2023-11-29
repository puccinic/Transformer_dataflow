#pragma once 
#include <cmath>

template<typename T, size_t size>
void softmax(T input[size], T result[size]) {
	T sum = 0;
	T tmp[size] = {};
	for (size_t i = 0; i < size; i++) {
		tmp[i] = std::exp(input[i]);
		sum += std::exp(input[i]);
	}

	for (size_t i = 0; i < size; i++) {
		result[i] = (T)((tmp[i] / sum));
	}
}

template<typename T, size_t size>
void masked_sofmax(T input[size], T mask[size], T result[size]) {
	T sum = 0;
	T tmp[size] = {};
softmax_sum_loop:
	for (size_t i = 0; i < size; i++) {
		tmp[i] = mask[i] ? std::exp(input[i]) : 0;
		sum += tmp[i];
	}
softmax_result_loop:
	for (size_t i = 0; i < size; i++) {
		result[i] = (T)((tmp[i] / sum));
	}
}