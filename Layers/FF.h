#pragma once
#include "Linear.h"
#include "Activations.h"

template<typename T, size_t rows, size_t hidden, size_t cols>
void ff(
	T input[rows][cols],
	T weights1[cols][hidden],
	T biases1[hidden],
	T weights2[hidden][cols],
	T biases2[cols],
	T result[rows][cols]
	) {
	T tmp1[rows][hidden];
	linear<T, rows, cols, hidden>(input, weights1, biases1, tmp1);
	T tmp2[rows][hidden];
	activation<T, rows, hidden>(tmp1, tmp2);
	linear<T, rows, hidden, cols>(tmp2, weights2, biases2, result);
}
