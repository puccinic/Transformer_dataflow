#pragma once
#include "Linear.h"
#include "Activations.h"

template<typename T, size_t rows, size_t hidden, size_t cols>
struct FF {
	Linear<T, rows, cols, hidden> linear1;
	Linear<T,rows,hidden,cols> linear2;
	
	void init(T weights1[cols][hidden], 
		T biases1[hidden], 
		T weights2[hidden][cols], 
		T biases2[cols]) {
		linear1.init(weights1,biases1);
		linear2.init(weights2, biases2);
	}

	void operator()(T input[rows][cols], T result[rows][cols]) {
		T tmp1[rows][hidden]{};
		linear1(input, tmp1);
		T tmp2[rows][hidden]{};
		activation<T, rows, hidden>(tmp1, tmp2, relu);
		linear2(tmp2, result);
	}
};
