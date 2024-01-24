#include "Definitions.h"
#include "Encoder.h"
#ifdef ENCODER
void accel(
	idata_t head_weights[NUM_HEADS][NUM_LINEAR_LAYERS][TOKEN_LEN][HEAD_LEN],
	idata_t head_biases[NUM_HEADS][NUM_LINEAR_LAYERS][HEAD_LEN],
	idata_t linear_weights[TOKEN_LEN][TOKEN_LEN],
	idata_t linear_bias[TOKEN_LEN],
	idata_t ff_weights1[TOKEN_LEN][HIDDEN],
	idata_t ff_biases1[HIDDEN],
	idata_t ff_weights2[HIDDEN][TOKEN_LEN],
	idata_t ff_biases2[TOKEN_LEN],
	idata_t gamma[NUM_LAYER_NORM][TOKEN_LEN],
	idata_t beta[NUM_LAYER_NORM][TOKEN_LEN],
	idata_t input[SEQ_LEN][TOKEN_LEN],
	idata_t input_mask[SEQ_LEN][SEQ_LEN],
	odata_t result[SEQ_LEN][TOKEN_LEN]
) {	
	idata_t epsilon[NUM_LAYER_NORM] = {EPSILON, EPSILON};
	encoder<idata_t, NUM_HEADS, SEQ_LEN, TOKEN_LEN, HEAD_LEN, HIDDEN>(
		input, 
		input_mask,
		head_weights,
		head_biases,
		linear_weights,
		linear_bias,
		ff_weights1,
		ff_biases1,
		ff_weights2,
		ff_biases2,
		epsilon,
		gamma,
		beta,
		result
	);
}
#endif

#ifdef MULTIHEAD
void accel(
	idata_t head_weights[NUM_HEADS][NUM_LINEAR_LAYERS][TOKEN_LEN][HEAD_LEN],
	idata_t head_biases[NUM_HEADS][NUM_LINEAR_LAYERS][HEAD_LEN],
	idata_t linear_weights[TOKEN_LEN][TOKEN_LEN],
	idata_t linear_bias[TOKEN_LEN],
	idata_t input[SEQ_LEN][TOKEN_LEN],
	idata_t input_mask[SEQ_LEN][SEQ_LEN],
	odata_t result[SEQ_LEN][SEQ_LEN]
) {
	multi_head_att<idata_t, NUM_HEADS, SEQ_LEN, TOKEN_LEN, HEAD_LEN> (
		input,
		input,
		input,
		input_mask,
		head_weights,
		head_biases,
		linear_weights,
		linear_bias,
		result
	);
}
#endif

#ifdef ATTHEAD
void accel(
	idata_t weights[NUM_LINEAR_LAYERS][TOKEN_LEN][HEAD_LEN],
	idata_t biases[NUM_LINEAR_LAYERS][HEAD_LEN],
	idata_t input[SEQ_LEN][TOKEN_LEN],
	idata_t input_mask[SEQ_LEN][SEQ_LEN],
	odata_t result[SEQ_LEN][TOKEN_LEN],
) {
	att_head<idata_t, SEQ_LEN, TOKEN_LEN, HEAD_LEN>(
		input,
		input,
		input,
		input_mask,
		weights,
		biases,
		result
	);
}
#endif

#ifdef FDFRWRD
void accel(
	idata_t weights1[SEQ_LEN][HIDDEN],
	idata_t biases1[HIDDEN],
	idata_t weights2[HIDDEN][TOKEN_LEN],
	idata_t biases2[TOKEN_LEN],
	idata_t input[SEQ_LEN][TOKEN_LEN],
	odata_t result[SEQ_LEN][TOKEN_LEN]
) {
	ff<idata_t, SEQ_LEN, HIDDEN, TOKEN_LEN>(
		input,
		weights1,
		biases1,
		weights2,
		biases2,
		result
	);
}
#endif

#ifdef LAYERNORM
void accel(
	idata_t gamma[TOKEN_LEN],
	idata_t beta[TOKEN_LEN],
	idata_t input[SEQ_LEN][TOKEN_LEN],
	odata_t result[SEQ_LEN][SEQ_LEN]
) {
	layer_norm<idata_t, SEQ_LEN, TOKEN_LEN>(
		input,
		epsilon,
		gamma,
		beta,
		result
	);
}
#endif

#ifdef DOTPRODATT
void accel(
	idata_t input[SEQ_LEN][TOKEN_LEN],
	idata_t input_mask[SEQ_LEN][SEQ_LEN],
	odata_t result[SEQ_LEN][TOKEN_LEN]
) {
	scaledotatt<idata_t, SEQ_LEN, TOKEN_LEN> (
		input, input, input,
		input_mask,
		result
	);
}
#endif

#ifdef LINEAR
void accel(
	idata_t weights[HIDDEN][TOKEN_LEN],
	idata_t biases[TOKEN_LEN],
	idata_t input[SEQ_LEN][HIDDEN],
	odata_t result[SEQ_LEN][TOKEN_LEN]
) {
	linear<idata_t, SEQ_LEN, HIDDEN, TOKEN_LEN> (
		input,
		weights,
		biases,
		result
	);
}
#endif

#ifdef MATMUL
void accel(
	idata_t matA[SEQ_LEN][HIDDEN],
	idata_t matB[HIDDEN][TOKEN_LEN],
	odata_t matRes[SEQ_LEN][TOKEN_LEN]
) {
	matmul<idata_t, SEQ_LEN, HIDDEN, TOKEN_LEN> (
		matA,
		matB,
		matRes
	);
}
#endif

#ifdef SOFTMAX
void accel(idata_t input[SEQ_LEN], idata_t result[SEQ_LEN]) {
	softmax<idata_t, SEQ_LEN>(input, result);
}
#endif

#ifdef ACTIVATION
void accel(
	idata_t input[SEQ_LEN][TOKEN_LEN],
	odata_t result[SEQ_LEN][TOKEN_LEN]
) {
	activation<idata_t, SEQ_LEN, TOKEN_LEN>(
		input,
		result
	);
}
#endif
