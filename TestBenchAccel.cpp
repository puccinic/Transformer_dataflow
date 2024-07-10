#include "Definitions.h"
#include "TestUtils.h"

enum
{
	MATIN,
	MATMASK,
	MATHEADWEIGHT,
	MATHEADBIAS,
	MATLINEARWEIGHT,
	MATLINEARBIAS,
	MATFFWEIGHTS1,
	MATFFBIAS1,
	MATFFWEIGHTS2,
	MATFFBIAS2,
	MATGAMMA,
	MATBETA,
	MATMEAN,
	MATVAR,
	MATNUM
};


int main(void)
{

	hls::stream<attention_weight_T> head_weights[NUM_HEADS][NUM_LINEAR_LAYERS][TOKEN_LEN];
	hls::stream<linear_weight_T> linear_weights[INNER_ATT_LINEAR_DIM];
	hls::stream<linear_bias_T> linear_bias[TOKEN_LEN];
	hls::stream<feedforward_weight1_T> ff_weights1[TOKEN_LEN];
	hls::stream<feedforward_weight2_T> ff_weights2[HIDDEN];
	hls::stream<feedforward_bias2_T> ff_biases2[TOKEN_LEN];
	hls::stream<gamma_T> gamma[NUM_LAYER_NORM][SEQ_LEN];
	hls::stream<beta_T> beta[NUM_LAYER_NORM][SEQ_LEN];
	hls::stream<mean_T> mean[NUM_LAYER_NORM][SEQ_LEN];
    hls::stream<variance_T> variance[NUM_LAYER_NORM][SEQ_LEN];
	hls::stream<input_T> input[TOKEN_LEN];
	hls::stream<result_T> result[TOKEN_LEN];

    std::string input_filename[MATNUM] =
	{
		[MATIN] = "../../../../input.txt",
		[MATMASK] = "../../../../mask.txt",
		[MATHEADWEIGHT] = "../../../../headweights.txt",
		[MATHEADBIAS] = "../../../../headbias.txt",
		[MATLINEARWEIGHT] = "../../../../linearweights.txt",
		[MATLINEARBIAS] = "../../../../linearbias.txt",
		[MATFFWEIGHTS1] = "../../../../ffweights1.txt",
		[MATFFBIAS1] = "../../../../ffbias1.txt",
		[MATFFWEIGHTS2] = "../../../../ffweights2.txt",
		[MATFFBIAS2] = "../../../../ffbias2.txt",
		[MATGAMMA] = "../../../../gamma.txt",
		[MATBETA] = "../../../../beta.txt",
		[MATMEAN] = "../../../../mean.txt",
		[MATVAR] = "../../../../variance.txt"
	};

	std::string result_filename = "../../../../golden_result.txt";
	std::string log_filename = "../../../../log.txt";

	load_stream_array<input_T, 1, SEQ_LEN, TOKEN_LEN>(&input, input_filename[MATIN]);

	load_stream_array<attention_weight_T, NUM_HEADS*NUM_LINEAR_LAYERS, HEAD_LEN, TOKEN_LEN>(head_weights[0], input_filename[MATHEADWEIGHT]);


	load_stream_array<linear_weight_T, 1, TOKEN_LEN, INNER_ATT_LINEAR_DIM>(&linear_weights, input_filename[MATLINEARWEIGHT]);

	load_stream_array<linear_bias_T, 1, 1, TOKEN_LEN>(&linear_bias, input_filename[MATLINEARBIAS]);

	load_stream_array<feedforward_weight1_T, 1, HIDDEN, TOKEN_LEN>(&ff_weights1, input_filename[MATFFWEIGHTS1]);


	load_stream_array<feedforward_weight2_T, 1, TOKEN_LEN, HIDDEN>(&ff_weights2, input_filename[MATFFWEIGHTS2]);

	load_stream_array<feedforward_bias2_T, 1, 1, TOKEN_LEN>(&ff_biases2, input_filename[MATFFBIAS2]);

	load_stream_array<gamma_T, NUM_LAYER_NORM, 1, SEQ_LEN>(gamma, input_filename[MATGAMMA]);

	load_stream_array<beta_T, NUM_LAYER_NORM, 1, SEQ_LEN>(beta, input_filename[MATBETA]);

	load_stream_array<mean_T, NUM_LAYER_NORM, 1, SEQ_LEN>(mean, input_filename[MATMEAN]);

	load_stream_array<variance_T, NUM_LAYER_NORM, 1, SEQ_LEN>(variance, input_filename[MATVAR]);

	accel(
		head_weights,
		linear_weights,
		linear_bias,
		ff_weights1,
		ff_weights2,
		ff_biases2,
		gamma,
		beta,
		mean,
		variance,
		input,
		result
	);

	compare_stream<result_T, SEQ_LEN, TOKEN_LEN>(result, &result_filename, &log_filename);
}
