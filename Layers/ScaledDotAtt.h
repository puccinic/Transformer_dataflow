#pragma once 
#include "MatMul.h"
#include "Transpose.h"
#include "Mask.h"
#include "SoftMax.h"
#include "Scale.h"

template<typename T, size_t sequence_length, size_t token_length>
void scaledotatt(T query[sequence_length][token_length],
	T key[sequence_length][token_length],
	T value[sequence_length][token_length],
	T input_mask[sequence_length][sequence_length],
	T result[sequence_length][token_length]) {
	
	constexpr T scale_factor = std::sqrt(token_length);

	T key_t[token_length][sequence_length]{};
	transpose_matrix<T, sequence_length, token_length>(key, key_t);

	T queryxkey[sequence_length][sequence_length]{};
	matmul<T, sequence_length, token_length, sequence_length>(query, key_t, queryxkey);

    T scaled_queryxkey[sequence_length][sequence_length]{};
	scale<T, sequence_length, sequence_length>(queryxkey, scaled_queryxkey, scale_factor);

	T softmax_att[sequence_length][sequence_length]{};
	for (size_t i = 0; i < sequence_length; i++) {
		masked_sofmax<T, sequence_length>(scaled_queryxkey[i], mask[i], softmax_att[i]);
	}

	matmul<T, sequence_length, sequence_length, token_length>(softmax_att, value, result);
}