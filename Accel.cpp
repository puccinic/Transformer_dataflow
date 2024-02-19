#include "Definitions.h"
#include "Encoder.h"

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
)
{
	idata_t epsilon[NUM_LAYER_NORM] = {EPSILON, EPSILON};
	encoder<idata_t, NUM_HEADS, SEQ_LEN, TOKEN_LEN, HEAD_LEN, HIDDEN>(
		input,
		input_mask,
		head_weights,
		head_biases,
		linear_weights,
		linear_bias,
		ff_weights1,
		ff_biases1,
		ff_weights2,
		ff_biases2,
		epsilon,
		gamma,
		beta,
	#if defined(USING_BATCH_NORM)
	    mean,
    	stddev,
	#endif /* using batch norm */
		result
	);
}
