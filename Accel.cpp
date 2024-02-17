#include "Definitions.h"
#include "Encoder.h"

void accel
(
	hls::stream<hls::vector<idata, token_length>> head_weights[num_heads][NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<idata, head_token_length>> head_biases[num_heads][NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<idata, token_length>> &linear_weights,
	hls::stream<hls::vector<idata, token_length>> &linear_bias,
	hls::stream<hls::vector<idata, token_length>> &ff_weights1,
	hls::stream<hls::vector<idata, hidden>> &ff_biases1,
	hls::stream<hls::vector<idata, hidden>> &ff_weights2,
	hls::stream<hls::vector<idata, token_length>> &ff_biases2,
	hls::stream<hls::vector<idata, token_length>> &gamma,
	hls::stream<hls::vector<idata, token_length>> &beta,
#if defined(USING_BATCH_NORM)
	hls::stream<hls::vector<idata, token_length>> &mean,
    hls::stream<hls::vector<idata, token_length>> &stddev,
#endif /* using batch norm */
	hls::stream<hls::vector<idata, token_length>> &input,
	hls::stream<hls::vector<idata, sequence_length>> &input_mask,
	hls::stream<hls::vector<odata_t, token_length>> &result
)
{
	idata_t epsilon[NUM_LAYER_NORM] = {EPSILON, EPSILON};
	encoder<idata_t, NUM_HEADS, SEQ_LEN, TOKEN_LEN, HEAD_LEN, HIDDEN>
	(
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
