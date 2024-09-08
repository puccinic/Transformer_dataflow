#include "Definitions.h"
#include "Encoder.h"

void accel(
	hls::stream<attention_weight_T> head_weights[NUM_LINEAR_LAYERS],
	hls::stream<linear_weight_T>& linear_weights,
	hls::stream<linear_bias_T>& linear_bias,
	hls::stream<feedforward_weight1_T>& ff_weights1,
	hls::stream<feedforward_weight2_T>& ff_weights2,
	hls::stream<feedforward_bias2_T>& ff_biases2,
	hls::stream<gamma_T> gamma[NUM_LAYER_NORM],
	hls::stream<beta_T> beta[NUM_LAYER_NORM],
	hls::stream<input_T>& input,
	hls::stream<result_T>& result

)
{
	encoder<
		input_T,
		attention_weight_T,
		linear_weight_T,
		linear_bias_T,
		feedforward_weight1_T,
		feedforward_weight2_T,
		feedforward_bias2_T,
		gamma_T,
		beta_T,
		norm_result1_T,
		attention_intermediate1_T,
		attention_intermediate2_T,
		attention_output_T,
		multi_head_attention_linear_intermediate_T,
		multi_head_attention_result_T,
		res_bock_T,
		norm_result2_T,
		feedforward_intermediate_T,
		feedforward_linear2_intermediate_T,
		feedforward_resutlt_T,
		result_T,
		SEQ_LEN,
		TOKEN_LEN,
		HEAD_LEN,
		HIDDEN
	>
	(
		input,
		head_weights,
		linear_weights,
		linear_bias,
		ff_weights1,
		ff_weights2,
		ff_biases2,
		gamma,
		beta,
		result
	);
}
