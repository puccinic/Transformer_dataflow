#pragma once

#include <hls_math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include "hls_vector.h"

/*Constants are set according to BERT-Tiny dimentions */

#define NUM_HEADS 2
#define SEQ_LEN 10
#define TOKEN_LEN 10
#define HEAD_LEN TOKEN_LEN / NUM_HEADS
#define HIDDEN 10
#define NUM_LINEAR_LAYERS 3
#define NUM_LAYER_NORM 2
#define SCALE_FACTOR 8
#define USING_BATCH_NORM
#define BITWIDTHI 16
#define	INTWIDTHI 4
#define BITWIDTHWH 16
#define INTWIDTHWH 4
#define BITWIDTHBH 16
#define INTWIDTHBH 4
#define BITWIDTHWL 16
#define INTWIDTHWL 4
#define BITWIDTHBL 16
#define INTWIDTHBL 4
#define BITWIDTHWFF1 16
#define INTWIDTHWFF1 4
#define BITWIDTHBFF1 16
#define INTWIDTHBFF1 4
#define BITWIDTHWFF2 16
#define INTWIDTHWFF2 4
#define BITWIDTHBFF2 16
#define INTWIDTHBFF2 4
#define BITWIDTHG 16
#define INTWIDTHG 4
#define BITWIDTHB 16
#define INTWIDTHB 4
#ifdef USING_BATCH_NORM
	#define BITWIDTHM 16
	#define INTWIDTHM 4
	#define BITWIDTHS 16
	#define INTWIDTHS 4
#endif /* using batch norm */
#define BITWIDTHR 16
#define INTWIDTHR 4
//#define USING_MASKED_SOFTMAX
void accel
(
	hls::stream<hls::vector<ap_fixed<BITWIDTHWH, INTWIDTHWH>, TOKEN_LEN>> head_weights[NUM_HEADS][NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<ap_fixed<BITWIDTHBH, INTWIDTHBH>, HEAD_LEN>> head_biases[NUM_HEADS][NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<ap_fixed<BITWIDTHWL, INTWIDTHWL>, TOKEN_LEN>> &linear_weights,
	hls::stream<hls::vector<ap_fixed<BITWIDTHBL, INTWIDTHBL>, TOKEN_LEN>> &linear_bias,
	hls::stream<hls::vector<ap_fixed<BITWIDTHWFF1, INTWIDTHWFF1>, TOKEN_LEN>> &ff_weights1,
	hls::stream<hls::vector<ap_fixed<BITWIDTHBFF1, INTWIDTHBFF1>, HIDDEN>> &ff_biases1,
	hls::stream<hls::vector<ap_fixed<BITWIDTHWFF2, INTWIDTHWFF2>, HIDDEN>> &ff_weights2,
	hls::stream<hls::vector<ap_fixed<BITWIDTHBFF2, INTWIDTHBFF2>, TOKEN_LEN>> &ff_biases2,
	hls::stream<hls::vector<ap_fixed<BITWIDTHG, INTWIDTHG>, TOKEN_LEN>> gamma[NUM_LAYER_NORM],
	hls::stream<hls::vector<ap_fixed<BITWIDTHB, INTWIDTHB>, TOKEN_LEN>> beta[NUM_LAYER_NORM],
#ifdef USING_BATCH_NORM
	hls::stream<hls::vector<ap_fixed<BITWIDTHM, INTWIDTHM>, TOKEN_LEN>> mean[NUM_LAYER_NORM],
    hls::stream<hls::vector<ap_fixed<BITWIDTHS, INTWIDTHS>, TOKEN_LEN>> stddev[NUM_LAYER_NORM],
#endif /* using batch norm */
	hls::stream<hls::vector<ap_fixed<BITWIDTHI, INTWIDTHI>, TOKEN_LEN>> &input,
#ifdef USING_MASKED_SOFTMAX
	hls::stream<hls::vector<bool, SEQ_LEN>> &input_mask,
#endif
	hls::stream<hls::vector<ap_fixed<BITWIDTHR, INTWIDTHR>, TOKEN_LEN>> &result
);
