#pragma once

#include <hls_math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include "hls_vector.h"

/*Constants are set according to BERT-Tiny dimentions */

#define EPSILON 0.0625
#define NUM_HEADS 2
#define SEQ_LEN 384
#define TOKEN_LEN 128
#define HEAD_LEN TOKEN_LEN / NUM_HEADS
#define HIDDEN 512
#define NUM_LINEAR_LAYERS 3
#define NUM_LAYER_NORM 2
#define SCALE_FACTOR 8
#define IN_WIDTH 8
#define IN_IWIDTH 4
#define OUT_WIDTH 8
#define OUT_IWIDTH 4
#define USING_APFIXED
#define USING_BATCH_NORM
#if defined(USING_APFIXED)

typedef ap_fixed<IN_WIDTH, IN_IWIDTH> odata_t;
typedef ap_fixed<IN_WIDTH, IN_IWIDTH> idata_t;

#else
	typedef float idata_t;
	typedef float odata_t;
#endif /*using ap_fixed */

void accel
(
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
#if defined(USING_BATCH_NORM)
	idata_t mean[NUM_LAYER_NORM][TOKEN_LEN],
	idata_t stddev[NUM_LAYER_NORM][TOKEN_LEN],
#endif /* using batch norm */
	idata_t input[SEQ_LEN][TOKEN_LEN],
	idata_t input_mask[SEQ_LEN][SEQ_LEN],
	odata_t result[SEQ_LEN][TOKEN_LEN]
);