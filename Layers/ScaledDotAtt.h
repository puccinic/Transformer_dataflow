#pragma once 
#include "MatMul.h"
#include "Transpose.h"
#include "Mask.h"
#include "SoftMax.h"
#include "Scale.h"

template<typename T, int sequence_length, int token_length>
void scaledotatt(
	T query[sequence_length][token_length],
	T key[sequence_length][token_length],
	T value[sequence_length][token_length],
	T input_mask[sequence_length][sequence_length],
	T result[sequence_length][token_length]
) {
	#pragma HLS DATAFLOW
	T softmax_att[sequence_length][sequence_length];
	matmul_scale_masked_softmax<T,sequence_length,token_length,sequence_length>(query, key, SCALE_FACTOR, input_mask, softmax_att);

	matmul<T, sequence_length, sequence_length, token_length>(softmax_att, value, result);
}
