#pragma once
#include "VecAdd.h"
#include "MatMul.h"

template<typename T, size_t rows, size_t hidden, size_t cols>
struct Linear {
	T (*weights)[cols];
	T* biases;

	void init(T weights[hidden][cols], T biases[cols]) {
		this->weights = weights;
		this->biases = biases;
	}

	void operator()(T input[rows][hidden], T result[rows][cols]) {
		T tmp[rows][cols]{};
		matmul<T, rows, hidden, cols>(input, weights, tmp);
		for (int i = 0; i < rows; i++) {
			vecadd<T, cols>(tmp[i], biases, result[i]);
		}
	}
};
