#pragma once 
#include "MatMul.h"
#include "Transpose.h"
#include "Mask.h"
#include "SoftMax.h"
#include "Scale.h"


/*sqrt Code taken from: https://stackoverflow.com/questions/8622256/in-c11-is-sqrt-defined-as-constexpr */

template<typename T>
T constexpr sqrtNewtonRaphson(T x, T curr, T prev) {
	return curr == prev
		? curr
		: sqrtNewtonRaphson(x, (curr + x / curr) / 2, curr);
}

/*
* Constexpr version of the square root
* Return value:
*   - For a finite and non-negative value of "x", returns an approximation for the square root of "x"
*   - Otherwise, returns NaN
*/
template<typename T>
double constexpr sqrt(T x) {
    return x >= 0 && x < std::numeric_limits<T>::infinity()
        ? sqrtNewtonRaphson<T>(x, x, 0)
        : std::numeric_limits<T>::quiet_NaN();
}

template<typename T, int sequence_length, int token_length>
void scaledotatt(
	T query[sequence_length][token_length],
	T key[sequence_length][token_length],
	T value[sequence_length][token_length],
	T input_mask[sequence_length][sequence_length],
	T result[sequence_length][token_length]
) {
	constexpr T scale_factor = sqrt<T>(token_length);
	
	T softmax_att[sequence_length][sequence_length];
	matmul_scale_masked_softmax<T,sequence_length,token_length,sequence_length>(query, key, scale_factor, input_mask, softmax_att);

	matmul<T, sequence_length, sequence_length, token_length>(softmax_att, value, result);
}
