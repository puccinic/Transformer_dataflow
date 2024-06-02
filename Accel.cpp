#include "Definitions.h"
#include "Encoder.h"

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
