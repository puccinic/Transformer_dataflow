#pragma once
#include <hls_math.h>
#include <ap_fixed.h>

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
#if defined(USING_APFIXED)

typedef ap_fixed<IN_WIDTH, IN_IWIDTH> odata_t;
typedef ap_fixed<IN_WIDTH, IN_IWIDTH> idata_t;

#else
	typedef float idata_t;
	typedef float odata_t;
#endif /*using ap_fixed */

#define ENCODER

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
);
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
);
#endif

#ifdef ATTHEAD
void accel(
	idata_t weights[NUM_LINEAR_LAYERS][TOKEN_LEN][HEAD_LEN],
	idata_t biases[NUM_LINEAR_LAYERS][HEAD_LEN],
	idata_t input[SEQ_LEN][TOKEN_LEN],
	idata_t input_mask[SEQ_LEN][SEQ_LEN],
	odata_t result[SEQ_LEN][TOKEN_LEN],
);
#endif

#ifdef FDFRWRD
void accel(
	idata_t weights1[SEQ_LEN][HIDDEN],
	idata_t biases1[HIDDEN],
	idata_t weights2[HIDDEN][TOKEN_LEN],
	idata_t biases2[TOKEN_LEN],
	idata_t input[SEQ_LEN][TOKEN_LEN],
	odata_t result[SEQ_LEN][TOKEN_LEN]
);
#endif

#ifdef LAYERNORM
void accel(
	idata_t gamma[TOKEN_LEN],
	idata_t beta[TOKEN_LEN],
	idata_t input[SEQ_LEN][TOKEN_LEN],
	odata_t result[SEQ_LEN][SEQ_LEN]
);
#endif

#ifdef DOTPRODATT
void accel(
	idata_t input[SEQ_LEN][TOKEN_LEN],
	idata_t input_mask[SEQ_LEN][SEQ_LEN],
	odata_t result[SEQ_LEN][TOKEN_LEN]
);
#endif

#ifdef LINEAR
void accel(
	idata_t weights[HIDDEN][TOKEN_LEN],
	idata_t biases[TOKEN_LEN],
	idata_t input[SEQ_LEN][HIDDEN],
	odata_t result[SEQ_LEN][TOKEN_LEN]
);
#endif

#ifdef MATMUL
void accel(
	idata_t matA[SEQ_LEN][HIDDEN],
	idata_t matB[HIDDEN][TOKEN_LEN],
	odata_t matRes[SEQ_LEN][TOKEN_LEN]
);
#endif

#ifdef SOFTMAX
void accel(idata_t input[SEQ_LEN], idata_t result[SEQ_LEN]);
#endif

#ifdef ACTIVATION
void accel(
	idata_t input[SEQ_LEN][TOKEN_LEN],
	odata_t result[SEQ_LEN][TOKEN_LEN]
);
#endif
