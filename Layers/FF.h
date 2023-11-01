#pragma once
#include "Linear.h"
#include "Activations.h"

template<typename T, size_t rows, size_t hidden, size_t cols>
void FF(T input[rows][cols], 
	T weights_layer1[cols][hidden], 
	T biases_layer1[hidden], 
	T weights_layer2[hidden][cols],
	T biases_layer2[cols],
	T result[rows][cols]) {

	T tmp1[rows][hidden]{};
	linear<T, rows, cols, hidden>(input, weights_layer1, biases_layer1, tmp1);
	
	T tmp2[rows][hidden]{};
	activation<T, rows, hidden>(tmp1, tmp2, relu);

	linear<T, rows, hidden, cols>(tmp2, weights_layer2, biases_layer2, result);
}