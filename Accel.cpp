#include "Definitions.h"
#include "Encoder.h"

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
#if defined(USING_BATCH_NORM)
	hls::stream<hls::vector<ap_fixed<BITWIDTHM, INTWIDTHM>, TOKEN_LEN>> mean[NUM_LAYER_NORM],
    hls::stream<hls::vector<ap_fixed<BITWIDTHS, INTWIDTHS>, TOKEN_LEN>> stddev[NUM_LAYER_NORM],
#endif /* using batch norm */
	hls::stream<hls::vector<ap_fixed<BITWIDTHI, INTWIDTHI>, TOKEN_LEN>> &input,
	hls::stream<hls::vector<int, SEQ_LEN>> &input_mask,
	hls::stream<hls::vector<ap_fixed<BITWIDTHR, INTWIDTHR>, TOKEN_LEN>> &result
)
{
	idata_t epsilon[NUM_LAYER_NORM] = {EPSILON, EPSILON};
	encoder<
		BITWIDTHI,
		INTWIDTHI,
		BITWIDTHWH,
		INTWIDTHWH,
		BITWIDTHBH,
		INTWIDTHBH,
		BITWIDTHWL,
		INTWIDTHWL,
		BITWIDTHBL,
		INTWIDTHBL,
		BITWIDTHWFF1,
		INTWIDTHWFF1,
		BITWIDTHBFF1,
		INTWIDTHBFF1,
		BITWIDTHWFF2,
		INTWIDTHWFF2,
		BITWIDTHBFF2,
		INTWIDTHBFF2,
		BITWIDTHG,
		INTWIDTHG,
		BITWIDTHB,
		INTWIDTHB,
	#if defined(USING_BATCH_NORM)
		BITWIDTHM,
		INTWIDTHM,
		BITWIDTHS,
		INTWIDTHS,
	#endif /* using batch norm */
		BITWIDTHR,
		INTWIDTHR,
		NUM_HEADS,
		SEQ_LEN,
		TOKEN_LEN,
		HEAD_LEN,
		HIDDEN
	>(
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
