#pragma once

#include <hls_math.h>
#include <ap_fixed.h>
#include "hls_stream.h"

#define NUM_LINEAR_LAYERS 3
#define NUM_LAYER_NORM 2


/*Model parameters */
#define NUM_HEADS 1
#define SEQ_LEN 65
#define TOKEN_LEN 512
#define INNER_ATT_LINEAR_DIM 384
#define HEAD_LEN INNER_ATT_LINEAR_DIM / NUM_HEADS
#define HIDDEN 256
#define SCALE_FACTOR 19.595918267231077


void accel
(
	hls::stream<float> head_weights[NUM_HEADS][NUM_LINEAR_LAYERS][TOKEN_LEN],
	hls::stream<float> linear_weights[INNER_ATT_LINEAR_DIM],
	hls::stream<float> linear_bias[TOKEN_LEN],
	hls::stream<float> ff_weights1[TOKEN_LEN],
	hls::stream<float> ff_biases1[HIDDEN],
	hls::stream<float> ff_weights2[HIDDEN],
	hls::stream<float> ff_biases2[TOKEN_LEN],
	hls::stream<float> gamma[NUM_LAYER_NORM][SEQ_LEN],
	hls::stream<float> beta[NUM_LAYER_NORM][SEQ_LEN],
	hls::stream<float> mean[NUM_LAYER_NORM][SEQ_LEN],
    hls::stream<float> variance[NUM_LAYER_NORM][SEQ_LEN],
	hls::stream<float> input[TOKEN_LEN],
	hls::stream<float> result[TOKEN_LEN]
);
