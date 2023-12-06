#include "TestUtils.h"
#include "Definitions.h"

int main() {
	std::cout << "running test" << std::endl;
    std::string input_filename[] = {
		"input1.txt",
		"input2.txt",
		"input3.txt",
		"input4.txt",
		"input5.txt",
		"input6.txt",
		"input7.txt",
		"input8.txt",
		"input9.txt",
		"input10.txt",
		"input11.txt",
		"input12.txt"
	};
	
	std::string result_filename = "golden_result.txt";
	std::string log_filename = "log.txt";

#ifdef ENCODER
    idata_t input[SEQ_LEN][TOKEN_LEN]{};
	load_arr<idata_t, SEQ_LEN*TOKEN_LEN>((idata_t*)input, &input_filename[0]);
	std::cout << "loaded  input" << std::endl;		

	idata_t input_mask[SEQ_LEN][SEQ_LEN]{};
	load_arr<idata_t, SEQ_LEN*SEQ_LEN>((idata_t*)input_mask, &input_filename[1]);

	idata_t head_weights[NUM_HEADS][NUM_LINEAR_LAYERS][TOKEN_LEN][HEAD_LEN]{};
	load_arr<idata_t, NUM_HEADS*NUM_LINEAR_LAYERS*TOKEN_LEN*HEAD_LEN>((idata_t*)head_weights, &input_filename[2]);

	idata_t head_biases[NUM_HEADS][NUM_LINEAR_LAYERS][HEAD_LEN]{};
	load_arr<idata_t, NUM_HEADS*NUM_LINEAR_LAYERS*HEAD_LEN>((idata_t*)head_biases, &input_filename[3]);

	idata_t linear_weights[TOKEN_LEN][TOKEN_LEN]{};
	load_arr<idata_t, TOKEN_LEN*TOKEN_LEN>((idata_t*)linear_weights, &input_filename[4]);

	idata_t linear_bias[TOKEN_LEN]{};
	load_arr<idata_t, TOKEN_LEN>((idata_t*)linear_bias, &input_filename[5]);

	idata_t ff_weights1[TOKEN_LEN][HIDDEN]{};
	load_arr<idata_t, TOKEN_LEN*HIDDEN>((idata_t*)ff_weights1, &input_filename[6]);

	idata_t ff_biases1[HIDDEN]{};
	load_arr<idata_t, HIDDEN>(ff_biases1, &input_filename[7]);

	idata_t ff_weights2[HIDDEN][TOKEN_LEN]{};
	load_arr<idata_t, HIDDEN*TOKEN_LEN>((idata_t*)ff_weights2, &input_filename[8]);

	idata_t ff_biases2[TOKEN_LEN]{};
	load_arr<idata_t, TOKEN_LEN>(ff_biases2, &input_filename[9]);

	idata_t gamma[NUM_LAYER_NORM][TOKEN_LEN]{};
	load_arr<idata_t, NUM_LAYER_NORM*TOKEN_LEN>((idata_t*)gamma, &input_filename[10]);
		
	idata_t beta[2][TOKEN_LEN]{};
	load_arr<idata_t, NUM_LAYER_NORM*TOKEN_LEN>((idata_t*)beta, &input_filename[11]);
	
	std::cout << "begining output" << std::endl;

	odata_t output[SEQ_LEN][TOKEN_LEN]{};
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
		input,
		input_mask,
		output
	);
	std::cout << "produced output" << std::endl;

	compare_mat<odata_t, SEQ_LEN, TOKEN_LEN>(output, &result_filename, &log_filename);

	std::cout << "test end" << std::endl;
#endif

}
