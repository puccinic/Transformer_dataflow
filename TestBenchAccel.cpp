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

	hls::stream<hls::vector<float, TOKEN_LEN>> head_weights[NUM_HEADS][NUM_LINEAR_LAYERS];
	hls::stream<hls::vector<float, INNER_ATT_LINEAR_DIM>> linear_weights("Linear_Weights");
	hls::stream<hls::vector<float, TOKEN_LEN>> linear_bias("Linear_Bias");
	hls::stream<hls::vector<float, TOKEN_LEN>> ff_weights1("FF_Weights1");
	hls::stream<hls::vector<float, HIDDEN>> ff_biases1("FF_Bias1");
	hls::stream<hls::vector<float, HIDDEN>> ff_weights2("FF_Weights2");
	hls::stream<hls::vector<float, TOKEN_LEN>> ff_biases2("FF_Bias2");
	hls::stream<hls::vector<float, SEQ_LEN>> gamma[NUM_LAYER_NORM];
	hls::stream<hls::vector<float, SEQ_LEN>> beta[NUM_LAYER_NORM];
	hls::stream<hls::vector<float, SEQ_LEN>> mean[NUM_LAYER_NORM];
    hls::stream<hls::vector<float, SEQ_LEN>> variance[NUM_LAYER_NORM];
	hls::stream<hls::vector<float, TOKEN_LEN>> input("Input");
	hls::stream<hls::vector<float, TOKEN_LEN>> result("Result");

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

	load_stream_array<float, 1, SEQ_LEN, TOKEN_LEN>(&input, input_filename[MATIN]);

	load_stream_array<float, NUM_HEADS*NUM_LINEAR_LAYERS, HEAD_LEN, TOKEN_LEN>(head_weights[0], input_filename[MATHEADWEIGHT]);


	load_stream_array<float, 1, TOKEN_LEN, INNER_ATT_LINEAR_DIM>(&linear_weights, input_filename[MATLINEARWEIGHT]);

	load_stream_array<float, 1, 1, TOKEN_LEN>(&linear_bias, input_filename[MATLINEARBIAS]);

	load_stream_array<float, 1, HIDDEN, TOKEN_LEN>(&ff_weights1, input_filename[MATFFWEIGHTS1]);

	load_stream_array<float, 1, 1, HIDDEN>(&ff_biases1, input_filename[MATFFBIAS1]);

	load_stream_array<float, 1, TOKEN_LEN, HIDDEN>(&ff_weights2, input_filename[MATFFWEIGHTS2]);

	load_stream_array<float, 1, 1, TOKEN_LEN>(&ff_biases2, input_filename[MATFFBIAS2]);

	load_stream_array<float, NUM_LAYER_NORM, 1, SEQ_LEN>(gamma, input_filename[MATGAMMA]);

	load_stream_array<float, NUM_LAYER_NORM, 1, SEQ_LEN>(beta, input_filename[MATBETA]);

	load_stream_array<float, NUM_LAYER_NORM, 1, SEQ_LEN>(mean, input_filename[MATMEAN]);

	load_stream_array<float, NUM_LAYER_NORM, 1, SEQ_LEN>(variance, input_filename[MATVAR]);

	accel(
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
		input,
		result
	);

	compare_stream<float, SEQ_LEN, TOKEN_LEN>(result, &result_filename, &log_filename);
}
