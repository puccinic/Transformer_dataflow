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

	hls::stream<hls::vector<ap_fixed<BITWIDTHWH, INTWIDTHWH>, TOKEN_LEN>> head_weights[NUM_HEADS][NUM_LINEAR_LAYERS];
	hls::stream<hls::vector<ap_fixed<BITWIDTHBH, INTWIDTHBH>, HEAD_LEN>> head_biases[NUM_HEADS][NUM_LINEAR_LAYERS];
	hls::stream<hls::vector<ap_fixed<BITWIDTHWL, INTWIDTHWL>, TOKEN_LEN>> linear_weights("Linear_Weights");
	hls::stream<hls::vector<ap_fixed<BITWIDTHBL, INTWIDTHBL>, TOKEN_LEN>> linear_bias("Linear_Bias");
	hls::stream<hls::vector<ap_fixed<BITWIDTHWFF1, INTWIDTHWFF1>, TOKEN_LEN>> ff_weights1("FF_Weights1");
	hls::stream<hls::vector<ap_fixed<BITWIDTHBFF1, INTWIDTHBFF1>, HIDDEN>> ff_biases1("FF_Bias1");
	hls::stream<hls::vector<ap_fixed<BITWIDTHWFF2, INTWIDTHWFF2>, HIDDEN>> ff_weights2("FF_Weights2");
	hls::stream<hls::vector<ap_fixed<BITWIDTHBFF2, INTWIDTHBFF2>, TOKEN_LEN>> ff_biases2("FF_Bias2");
	hls::stream<hls::vector<ap_fixed<BITWIDTHG, INTWIDTHG>, TOKEN_LEN>> gamma[NUM_LAYER_NORM];
	hls::stream<hls::vector<ap_fixed<BITWIDTHB, INTWIDTHB>, TOKEN_LEN>> beta[NUM_LAYER_NORM];
#if defined(USING_BATCH_NORM)
	hls::stream<hls::vector<ap_fixed<BITWIDTHM, INTWIDTHM>, TOKEN_LEN>> mean[NUM_LAYER_NORM];
    hls::stream<hls::vector<ap_fixed<BITWIDTHS, INTWIDTHS>, TOKEN_LEN>> stddev[NUM_LAYER_NORM];
#endif /* using batch norm */
	hls::stream<hls::vector<ap_fixed<BITWIDTHI, INTWIDTHI>, TOKEN_LEN>> input("Input");
	hls::stream<hls::vector<idata_t, SEQ_LEN>> input_mask("Mask");
	hls::stream<hls::vector<ap_fixed<BITWIDTHR, INTWIDTHR>, TOKEN_LEN>> result("Result");

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

	load_stream_array<ap_fixed<BITWIDTHI, INTWIDTHI>, 1, SEQ_LEN, TOKEN_LEN>(&input, input_filename[MATIN]);

	load_stream_array<int, 1, SEQ_LEN, SEQ_LEN>(&input_mask, input_filename[MATMASK]);


	load_stream_array<ap_fixed<BITWIDTHWH, INTWIDTHWH>, NUM_HEADS*NUM_LINEAR_LAYERS, HEAD_LEN, TOKEN_LEN>(head_weights[0], input_filename[MATHEADWEIGHT]);

	load_stream_array<ap_fixed<BITWIDTHBH, INTWIDTHBH>, NUM_HEADS*NUM_LINEAR_LAYERS, 1, HEAD_LEN>(head_biases[0], input_filename[MATHEADBIAS]);

	load_stream_array<ap_fixed<BITWIDTHWL, INTWIDTHWL>, 1, TOKEN_LEN, TOKEN_LEN>(&linear_weights, input_filename[MATLINEARWEIGHT]);

	load_stream_array<ap_fixed<BITWIDTHBL, INTWIDTHBL>, 1, 1, TOKEN_LEN>(&linear_bias, input_filename[MATLINEARBIAS]);

	load_stream_array<ap_fixed<BITWIDTHWFF1, INTWIDTHWFF1>, 1, HIDDEN, TOKEN_LEN>(&ff_weights1, input_filename[MATFFWEIGHTS1]);

	load_stream_array<ap_fixed<BITWIDTHBFF1, INTWIDTHBFF1>, 1, 1, HIDDEN>(&ff_biases1, input_filename[MATFFBIAS1]);

	load_stream_array<ap_fixed<BITWIDTHWFF2, INTWIDTHWFF2>, 1, TOKEN_LEN, HIDDEN>(&ff_weights2, input_filename[MATFFWEIGHTS2]);

	load_stream_array<ap_fixed<BITWIDTHBFF2, INTWIDTHBFF2>, 1, 1, TOKEN_LEN>(&ff_biases2, input_filename[MATFFBIAS2]);

	load_stream_array<ap_fixed<BITWIDTHG, INTWIDTHG>, NUM_LAYER_NORM, 1, TOKEN_LEN>(gamma, input_filename[MATGAMMA]);

	load_stream_array<ap_fixed<BITWIDTHB, INTWIDTHB>, NUM_LAYER_NORM, 1, TOKEN_LEN>(beta, input_filename[MATBETA]);

#if defined(USING_BATCH_NORM)
	load_stream_array<ap_fixed<BITWIDTHM, INTWIDTHM>, NUM_LAYER_NORM, 1, TOKEN_LEN>(mean, input_filename[MATGAMMA]);

	load_stream_array<ap_fixed<BITWIDTHS, INTWIDTHS>, NUM_LAYER_NORM, 1, TOKEN_LEN>(stddev, input_filename[MATBETA]);
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

	compare_stream<ap_fixed<BITWIDTHR, INTWIDTHR>, SEQ_LEN, TOKEN_LEN>(result, &result_filename, &log_filename);
}
