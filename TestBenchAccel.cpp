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
	MATNUM
};


int main(void)
{

	hls::stream<hls::vector<idata_t, TOKEN_LEN>> head_weights[NUM_HEADS][NUM_LINEAR_LAYERS];
	hls::stream<hls::vector<idata_t, HEAD_LEN>> head_biases[NUM_HEADS][NUM_LINEAR_LAYERS];
	hls::stream<hls::vector<idata_t, TOKEN_LEN>> linear_weights;
	hls::stream<hls::vector<idata_t, TOKEN_LEN>> linear_bias;
	hls::stream<hls::vector<idata_t, TOKEN_LEN>> ff_weights1;
	hls::stream<hls::vector<idata_t, HIDDEN>> ff_biases1;
	hls::stream<hls::vector<idata_t, HIDDEN>> ff_weights2;
	hls::stream<hls::vector<idata_t, TOKEN_LEN>> ff_biases2;
	hls::stream<hls::vector<idata_t, TOKEN_LEN>> gamma[NUM_LAYER_NORM];
	hls::stream<hls::vector<idata_t, TOKEN_LEN>> beta[NUM_LAYER_NORM];
#if defined(USING_BATCH_NORM)
	hls::stream<hls::vector<idata_t, TOKEN_LEN>> mean[NUM_LAYER_NORM];
    hls::stream<hls::vector<idata_t, TOKEN_LEN>> stddev[NUM_LAYER_NORM];
#endif /* using batch norm */
	hls::stream<hls::vector<idata_t, TOKEN_LEN>> input;
	hls::stream<hls::vector<idata_t, SEQ_LEN>> input_mask;
	hls::stream<hls::vector<odata_t, TOKEN_LEN>> result;

    std::string input_filename[MATNUM] =
	{
		[MATIN] = "/home/carlos/Transformer_dataflow/input.txt",
		[MATMASK] = "/home/carlos/Transformer_dataflow/mask.txt",
		[MATHEADWEIGHT] = "/home/carlos/Transformer_dataflow/headweights.txt",
		[MATHEADBIAS] = "/home/carlos/Transformer_dataflow/headbias.txt",
		[MATLINEARWEIGHT] = "/home/carlos/Transformer_dataflow/linearweights.txt",
		[MATLINEARBIAS] = "/home/carlos/Transformer_dataflow/linearbias.txt",
		[MATFFWEIGHTS1] = "/home/carlos/Transformer_dataflow/ffweights1.txt",
		[MATFFBIAS1] = "/home/carlos/Transformer_dataflow/ffbias1.txt",
		[MATFFWEIGHTS2] = "/home/carlos/Transformer_dataflow/ffweights2.txt",
		[MATFFBIAS2] = "/home/carlos/Transformer_dataflow/ffbias2.txt",
		[MATGAMMA] = "/home/carlos/Transformer_dataflow/gamma.txt",
		[MATBETA] = "/home/carlos/Transformer_dataflow/beta.txt"
	};

	std::string result_filename = "/home/carlos/Transformer_dataflow/golden_result.txt";
	std::string log_filename = "/home/carlos/Transformer_dataflow/log.txt";

	load_stream_array<idata_t, 1, SEQ_LEN, TOKEN_LEN>(&input, input_filename[MATIN]);

	load_stream_array<idata_t, 1, SEQ_LEN, SEQ_LEN>(&input_mask, input_filename[MATMASK]);


	load_stream_array<idata_t, NUM_HEADS*NUM_LINEAR_LAYERS, HEAD_LEN, TOKEN_LEN>(head_weights[0], input_filename[MATHEADWEIGHT]);

	load_stream_array<idata_t, NUM_HEADS*NUM_LINEAR_LAYERS, 1, HEAD_LEN>(head_biases[0], input_filename[MATHEADBIAS]);

	load_stream_array<idata_t, 1, TOKEN_LEN, TOKEN_LEN>(&linear_weights, input_filename[MATLINEARWEIGHT]);

	load_stream_array<idata_t, 1, 1, TOKEN_LEN>(&linear_bias, input_filename[MATLINEARBIAS]);

	load_stream_array<idata_t, 1, HIDDEN, TOKEN_LEN>(&ff_weights1, input_filename[MATFFWEIGHTS1]);

	load_stream_array<idata_t, 1, 1, HIDDEN>(&ff_biases1, input_filename[MATFFBIAS1]);

	load_stream_array<idata_t, 1, TOKEN_LEN, HIDDEN>(&ff_weights2, input_filename[MATFFWEIGHTS2]);

	load_stream_array<idata_t, 1, 1, TOKEN_LEN>(&ff_biases2, input_filename[MATFFBIAS2]);

	load_stream_array<idata_t, NUM_LAYER_NORM, 1, TOKEN_LEN>(gamma, input_filename[MATGAMMA]);

	load_stream_array<idata_t, NUM_LAYER_NORM, 1, TOKEN_LEN>(beta, input_filename[MATBETA]);

#if defined(USING_BATCH_NORM)
	load_stream_array<idata_t, NUM_LAYER_NORM, 1, TOKEN_LEN>(mean, input_filename[MATGAMMA]);

	load_stream_array<idata_t, NUM_LAYER_NORM, 1, TOKEN_LEN>(stddev, input_filename[MATBETA]);
#endif /* using batch norm */

	accel(
		head_weights,
		head_biases,
		linear_weights,
		linear_bias,
		ff_weights1,
		ff_biases1,
		ff_weights2,
		ff_biases2,
		gamma,
		beta,
	#if defined(USING_BATCH_NORM)
		mean,
		stddev,
	#endif
		input,
		input_mask,
		result
	);

	compare_stream<odata_t, SEQ_LEN, TOKEN_LEN>(result, &result_filename, &log_filename);
}
