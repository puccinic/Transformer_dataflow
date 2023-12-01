#pragma once
#include "VecAdd.h"
#include "MatMul.h"

template<typename T, int rows, int hidden, int cols>
void linear(
	T input[rows][hidden],
	T weights[hidden][cols], 
	T biases[cols],
	T result[rows][cols]
	) {
	T tmp[rows][cols];
	matmul<T, rows, hidden, cols>(input, weights, tmp);
	for (int i = 0; i < rows; i++) {
		vecadd<T, cols>(tmp[i], biases, result[i]);
	}
}
