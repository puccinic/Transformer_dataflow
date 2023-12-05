#pragma once
#include <ap_fixed.h>
#include <hls_math.h>
#define EPSILON 1e-5
#define NUM_HEADS 1
#define SEQ_LEN 10
#define TOKEN_LEN 10
#define HEAD_LEN TOKEN_LEN / NUM_HEADS
#define HIDDEN 10
#define NUM_LINEAR_LAYERS 3
#define NUM_LAYER_NORM 2
#define SCALE_FACTOR 3.162277660168379
typedef int idata_t;
typedef int odata_t;

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
);

