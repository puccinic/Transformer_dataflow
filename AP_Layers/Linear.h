#pragma once
#include "AP_Utils.h"
#include "VecAdd.h"
#include "MatMul.h"
template<int width, int i_width, int rows, int hidden, int cols>
void linear(
	ap_fixed<width, i_width> input[rows][hidden],
	ap_fixed<width, i_width> weights[hidden][cols], 
	ap_fixed<width, i_width> biases[cols],
	ap_fixed<width, i_width> result[rows][cols]
) {
	ap_fixed<2*width + hidden, 2*i_width> tmp1[rows][cols];
	matmul<width, i_width, rows, hidden, cols>(input, weights, tmp1);
	
	ap_fixed<width, i_width> tmp2[rows][cols];
	ap_fixed_resize<2*width + hidden, 2*i_width, width, i_width>(tmp1, tmp2);

	ap_fixed<width + 1, i_width> tmp3[rows][cols];
	for (int i = 0; i < rows; i++) {
		vecadd<width, i_width, cols>(tmp2[i], biases, tmp3[i]);
	}

	ap_fixed_resize<width + 1, i_width, width, i_width>(tmp3, result);
}