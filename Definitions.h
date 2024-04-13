#pragma once

#include <hls_math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include "hls_vector.h"

/*Constants are set according to BERT-Tiny dimentions */

#define EPSILON 0.0625
#define NUM_HEADS 2
#define SEQ_LEN 10
#define TOKEN_LEN 10
#define HEAD_LEN TOKEN_LEN / NUM_HEADS
#define HIDDEN 10
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
	typedef double idata_t;
	typedef double odata_t;
#endif /*using ap_fixed */

void accel
(
	hls::stream<hls::vector<idata_t, TOKEN_LEN>> head_weights[NUM_HEADS][NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<idata_t, HEAD_LEN>> head_biases[NUM_HEADS][NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<idata_t, TOKEN_LEN>> &linear_weights,
	hls::stream<hls::vector<idata_t, TOKEN_LEN>> &linear_bias,
	hls::stream<hls::vector<idata_t, TOKEN_LEN>> &ff_weights1,
	hls::stream<hls::vector<idata_t, HIDDEN>> &ff_biases1,
	hls::stream<hls::vector<idata_t, HIDDEN>> &ff_weights2,
	hls::stream<hls::vector<idata_t, TOKEN_LEN>> &ff_biases2,
	hls::stream<hls::vector<idata_t, TOKEN_LEN>> gamma[NUM_LAYER_NORM],
	hls::stream<hls::vector<idata_t, TOKEN_LEN>> beta[NUM_LAYER_NORM],
#if defined(USING_BATCH_NORM)
	hls::stream<hls::vector<idata_t, TOKEN_LEN>> mean[NUM_LAYER_NORM],
    hls::stream<hls::vector<idata_t, TOKEN_LEN>> stddev[NUM_LAYER_NORM],
#endif /* using batch norm */
	hls::stream<hls::vector<idata_t, TOKEN_LEN>> &input,
	hls::stream<hls::vector<idata_t, SEQ_LEN>> &input_mask,
	hls::stream<hls::vector<odata_t, TOKEN_LEN>> &result
);
