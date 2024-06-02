#pragma once

#include <hls_math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include "hls_vector.h"

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
	hls::stream<hls::vector<float, TOKEN_LEN>> head_weights[NUM_HEADS][NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<float, INNER_ATT_LINEAR_DIM>> &linear_weights,
	hls::stream<hls::vector<float, TOKEN_LEN>> &linear_bias,
	hls::stream<hls::vector<float, TOKEN_LEN>> &ff_weights1,
	hls::stream<hls::vector<float, HIDDEN>> &ff_biases1,
	hls::stream<hls::vector<float, HIDDEN>> &ff_weights2,
	hls::stream<hls::vector<float, TOKEN_LEN>> &ff_biases2,
	hls::stream<hls::vector<float, SEQ_LEN>> gamma[NUM_LAYER_NORM],
	hls::stream<hls::vector<float, SEQ_LEN>> beta[NUM_LAYER_NORM],
	hls::stream<hls::vector<float, SEQ_LEN>> mean[NUM_LAYER_NORM],
    hls::stream<hls::vector<float, SEQ_LEN>> variance[NUM_LAYER_NORM],
	hls::stream<hls::vector<float, TOKEN_LEN>> &input,
	hls::stream<hls::vector<float, TOKEN_LEN>> &result
);
