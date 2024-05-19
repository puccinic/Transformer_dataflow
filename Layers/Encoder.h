#pragma once

#include "hls_stream.h"
#include "hls_vector.h"
#include "MultiHeadAtt.h"
#include "MatAdd.h"
#include "LayerNorm.h"
#include "FF.h"
#include "Synth_utils.h"

template
<
	int bitWidthI,
	int intWidthI,
	int bitWidthWH,
	int intWidthWH,
	int bitWidthBH,
	int intWidthBH,
	int bitWidthWL,
	int intWidthWL,
	int bitWidthBL,
	int intWidthBL,
	int bitWidthWFF1,
	int intWidthWFF1,
	int bitWidthBFF1,
	int intWidthBFF1,
	int bitWidthWFF2,
	int intWidthWFF2,
	int bitWidthBFF2,
	int intWidthBFF2,
	int bitWidthG,
	int intWidthG,
	int bitWidthB,
	int intWidthB,
#ifdef USING_BATCH_NORM
	int bitWidthM,
	int intWidthM,
	int bitWidthS,
	int intWidthS,
#endif /* using batch norm */
	int bitWidthR,
	int intWidthR,
	int num_heads,
	int sequence_length,
	int token_length,
	int head_token_length,
	int hidden
>
void encoder(
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, token_length>> &input,
#ifdef USING_MASKED_SOFTMAX
	hls::stream<hls::vector<bool, sequence_length>> &input_mask,
#endif
	hls::stream<hls::vector<ap_fixed<bitWidthWH, intWidthWH>, token_length>> head_weights[num_heads][NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<ap_fixed<bitWidthBH, intWidthBH>, head_token_length>> head_biases[num_heads][NUM_LINEAR_LAYERS],
	hls::stream<hls::vector<ap_fixed<bitWidthWL, intWidthWL>, token_length>> &linear_weights,
	hls::stream<hls::vector<ap_fixed<bitWidthBL, intWidthBL>, token_length>> &linear_bias,
	hls::stream<hls::vector<ap_fixed<bitWidthWFF1, intWidthWFF1>, token_length>> &ff_weights1,
	hls::stream<hls::vector<ap_fixed<bitWidthBFF1, intWidthBFF1>, hidden>> &ff_biases1,
	hls::stream<hls::vector<ap_fixed<bitWidthWFF2, intWidthWFF2>, hidden>> &ff_weights2,
	hls::stream<hls::vector<ap_fixed<bitWidthBFF2, intWidthBFF2>, token_length>> &ff_biases2,
	ap_fixed<bitWidthR, intWidthR> epsilon[NUM_LAYER_NORM],
	hls::stream<hls::vector<ap_fixed<bitWidthG, intWidthG>, token_length>> gamma[NUM_LAYER_NORM],
	hls::stream<hls::vector<ap_fixed<bitWidthB, intWidthB>, token_length>> beta[NUM_LAYER_NORM],
#ifdef USING_BATCH_NORM
	hls::stream<hls::vector<ap_fixed<bitWidthM, intWidthM>, token_length>> mean[NUM_LAYER_NORM],
    hls::stream<hls::vector<ap_fixed<bitWidthS, intWidthS>, token_length>> stddev[NUM_LAYER_NORM],
#endif /* using batch norm */
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, token_length>> &result
)
{
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, token_length>, sequence_length> input_copy1("input_copy1");
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, token_length>, sequence_length> input_copy2("input_copy2");
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, token_length>, sequence_length> input_copy3("input_copy3");
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, token_length>, sequence_length> input_copy4("input_copy4");
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, token_length>, sequence_length> multi_head_result("multi_head_res");
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, token_length>, sequence_length> matadd_result1("matadd_res1");
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, token_length>, sequence_length> norm_result("norm_res");
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, token_length>, sequence_length> norm_result_copy1("norm_res_copy1");
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, token_length>, sequence_length> norm_result_copy2("norm_res_copy2");
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, token_length>, sequence_length> ff_result("ff_res");
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, token_length>, sequence_length> matadd_result2("matadd_res2");

	#pragma HLS DATAFLOW
	replicate4<ap_fixed<bitWidthI, intWidthI>, sequence_length, token_length>(input, input_copy1, input_copy2, input_copy3, input_copy4);
	multi_head_att<bitWidthI, intWidthI, bitWidthWH, intWidthWH, bitWidthBH, intWidthBH, bitWidthWL, intWidthWL, bitWidthBL, intWidthBL, bitWidthR, intWidthR, num_heads, sequence_length, token_length, head_token_length>(
		input_copy1,
		input_copy2,
		input_copy3,
	#ifdef USING_MASKED_SOFTMAX
		input_mask,
	#endif
		head_weights,
		head_biases,
		linear_weights,
		linear_bias,
		multi_head_result
	);

	matadd<bitWidthI, intWidthI, bitWidthR, intWidthR, bitWidthR, intWidthR, sequence_length, token_length>(
		input_copy4,
		multi_head_result,
		matadd_result1
	);

#ifdef USING_BATCH_NORM
	batch_norm<bitWidthR, intWidthR, bitWidthG, intWidthG, bitWidthB, intWidthB, bitWidthM, intWidthM, bitWidthS, intWidthS, bitWidthR, intWidthR, sequence_length, token_length>(
		matadd_result1,
		epsilon[0],
		gamma[0],
		beta[0],
		mean[0],
		stddev[0],
		norm_result
	);
#else
	layer_norm<bitWidthR, intWidthR, bitWidthG, intWidthG, bitWidthB, intWidthB, bitWidthR, intWidthR, sequence_length, token_length>(
		matadd_result1,
		epsilon[0],
		gamma[0],
		beta[0],
		norm_result
	);
#endif /* using batch norm */

	replicate2<ap_fixed<bitWidthR, intWidthR>, sequence_length, token_length>(
		norm_result,
		norm_result_copy1,
		norm_result_copy2
	);

	ff<bitWidthR, intWidthR, bitWidthWFF1, intWidthWFF1, bitWidthBFF1, intWidthBFF1, bitWidthWFF2, intWidthWFF2, bitWidthBFF2, intWidthBFF2, bitWidthR, intWidthR, sequence_length, hidden, token_length>(
		norm_result_copy1,
		ff_weights1,
		ff_biases1,
		ff_weights2,
		ff_biases2,
		ff_result
	);

	matadd<bitWidthR, intWidthR, bitWidthR, intWidthR, bitWidthR, intWidthR, sequence_length, token_length>(
		norm_result_copy2,
		ff_result,
		matadd_result2
	);

#ifdef USING_BATCH_NORM
	batch_norm<bitWidthR, intWidthR, bitWidthG, intWidthG, bitWidthB, intWidthB, bitWidthM, intWidthM, bitWidthS, intWidthS, bitWidthR, intWidthR, sequence_length, token_length>(
		matadd_result2,
		epsilon[1],
		gamma[1],
		beta[1],
		mean[1],
		stddev[1],
		result
	);
#else
	layer_norm<bitWidthR, intWidthR, bitWidthG, intWidthG, bitWidthB, intWidthB, bitWidthR, intWidthR, sequence_length, token_length>(
		matadd_result2,
		epsilon[1],
		gamma[1],
		beta[1],
		result
	);
#endif /* using batch norm */
}
