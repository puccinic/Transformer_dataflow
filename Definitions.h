#pragma once

#include <hls_math.h>
#include <ap_fixed.h>
#include "hls_stream.h"

#define NUM_LINEAR_LAYERS 3
#define NUM_LAYER_NORM 2


/*Model parameters */
#define NUM_HEADS (1)
#define SEQ_LEN (65)
#define TOKEN_LEN (512)
#define INNER_ATT_LINEAR_DIM (384)
#define HEAD_LEN (INNER_ATT_LINEAR_DIM / NUM_HEADS)
#define HIDDEN (256)
#define SCALE_FACTOR (19.595918267231077)

typedef ap_fixed<8,6> input_T;
typedef ap_fixed<8,5> attention_weight_T;
typedef ap_fixed<8,4> linear_weight_T;
typedef ap_fixed<8,3> linear_bias_T;
typedef ap_fixed<8,2> feedforward_weight1_T;
typedef ap_fixed<8,1> feedforward_bias1_T;
typedef ap_fixed<8,4> feedforward_weight2_T;
typedef ap_fixed<8,5> feedforward_bias2_T;
typedef float gamma_T;
typedef float beta_T;
typedef float mean_T;
typedef float variance_T;
typedef ap_fixed<8,6> norm_result1_T;
typedef ap_fixed<8,7> attention_intermediate1_T;
typedef ap_fixed<8,4> attention_intermediate2_T;
typedef ap_fixed<8,3> attention_output_T;
typedef ap_fixed<8,5> multi_head_attention_linear_intermediate_T;
typedef ap_fixed<8,4> multi_head_attention_result_T;
typedef ap_fixed<8,6> res_bock_T;
typedef ap_fixed<8,7> norm_result2_T;
typedef ap_fixed<8,4> feedforward_linear1_intermediate_T;
typedef ap_fixed<8,3> feedforward_intermediate_T;
typedef ap_fixed<8,5> feedforward_linear2_intermediate_T;
typedef ap_fixed<8,6> feedforward_resutlt_T;
typedef ap_fixed<8,4> result_T;

void accel(
	hls::stream<attention_weight_T> head_weights[NUM_HEADS][NUM_LINEAR_LAYERS][TOKEN_LEN],
	hls::stream<linear_weight_T> linear_weights[INNER_ATT_LINEAR_DIM],
	hls::stream<linear_bias_T> linear_bias[TOKEN_LEN],
	hls::stream<feedforward_weight1_T> ff_weights1[TOKEN_LEN],
	hls::stream<feedforward_weight2_T> ff_weights2[HIDDEN],
	hls::stream<feedforward_bias2_T> ff_biases2[TOKEN_LEN],
	hls::stream<gamma_T> gamma[NUM_LAYER_NORM][SEQ_LEN],
	hls::stream<beta_T> beta[NUM_LAYER_NORM][SEQ_LEN],
	hls::stream<mean_T> mean[NUM_LAYER_NORM][SEQ_LEN],
    hls::stream<variance_T> variance[NUM_LAYER_NORM][SEQ_LEN],
	hls::stream<input_T> input[TOKEN_LEN],
	hls::stream<result_T> result[TOKEN_LEN]
);
