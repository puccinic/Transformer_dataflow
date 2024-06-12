#include "Definitions.h"
#include "Encoder.h"

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
)
{
	encoder<float, NUM_HEADS, SEQ_LEN, TOKEN_LEN, HEAD_LEN, HIDDEN>(
		input,
		head_weights,
		linear_weights,
		linear_bias,
		ff_weights1,
		ff_biases1,
		ff_weights2,
		ff_biases2,
		gamma,
		beta,
	    mean,
    	variance,
		result
	);
}
