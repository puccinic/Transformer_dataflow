#pragma once 
#include "MatMul.h"
#include "Transpose.h"
#include "Mask.h"
#include "SoftMax.h"
#include "Scale.h"


/*sqrt_helprt Code taken from: https://stackoverflow.com/questions/8622256/in-c11-is-sqrt-defined-as-constexpr */
template <typename T>
constexpr T sqrt_helper(T x, T lo, T hi)
{
  if (lo == hi)
    return lo;

  const T mid = (lo + hi + 1) / 2;

  if (x / mid < mid)
    return sqrt_helper<T>(x, lo, mid - 1);
  else
    return sqrt_helper(x, mid, hi);
}

template <typename T>
constexpr T ct_sqrt(T x)
{
  return sqrt_helper<T>(x, 0, x / 2 + 1);
}

template<typename T, int sequence_length, int token_length>
void scaledotatt(
	T query[sequence_length][token_length],
	T key[sequence_length][token_length],
	T value[sequence_length][token_length],
	T input_mask[sequence_length][sequence_length],
	T result[sequence_length][token_length]
) {
	constexpr T scale_factor = ct_sqrt<T>(token_length);
	
    T scaled_queryxkey[sequence_length][sequence_length];
	matmul_transpose_scale<T, sequence_length, token_length, sequence_length>(query, key, scale_factor, scaled_queryxkey);

	T softmax_att[sequence_length][sequence_length];
scaledotatt_loop:
	for (int i = 0; i < sequence_length; i++) {
		masked_sofmax<T, sequence_length>(scaled_queryxkey[i], input_mask[i], softmax_att[i]);
	}

	matmul<T, sequence_length, sequence_length, token_length>(softmax_att, value, result);
}
