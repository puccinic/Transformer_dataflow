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

    idata_t input[sequence_length][token_length];
	load_arr<idata_t, sequence_length*token_length>((idata_t*)input, &input_filename[0]);
		
	idata_t input_mask[sequence_length][sequence_length];
	load_arr<idata_t, sequence_length*sequence_length>((idata_t*)input_mask, &input_filename[1]);

	idata_t head_weights[num_heads][num_linear_layers][token_length][head_token_length];
	load_arr<idata_t, num_heads*num_linear_layers*token_length*head_token_length>((idata_t*)head_weights, &input_filename[2]);

	idata_t head_biases[num_heads][num_linear_layers][head_token_length];
	load_arr<idata_t, num_heads*num_linear_layers*head_token_length>((idata_t*)head_biases, &input_filename[3]);

	idata_t linear_weights[token_length][token_length];
	load_arr<idata_t, token_length*token_length>((idata_t*)linear_weights, &input_filename[4]);

	idata_t linear_bias[token_length];
	load_arr<idata_t, token_length>((idata_t*)linear_bias, &input_filename[5]);

	idata_t ff_weights1[token_length][hidden];
	load_arr<idata_t, token_length*hidden>((idata_t*)ff_weights1, &input_filename[6]);

	idata_t ff_biases1[hidden];
	load_arr<idata_t, hidden>(ff_biases1, &input_filename[7]);

	idata_t ff_weights2[hidden][token_length];
	load_arr<idata_t, hidden*token_length>((idata_t*)ff_weights2, &input_filename[8]);

	idata_t ff_biases2[token_length];
	load_arr<idata_t, token_length>(ff_biases2, &input_filename[9]);

	idata_t gamma[2][token_length];
	load_arr<idata_t, 2*token_length>((idata_t*)gamma, &input_filename[10]);
		
	idata_t beta[2][token_length];
	load_arr<idata_t, 2*token_length>((idata_t*)beta, &input_filename[11]);

	odata_t output[sequence_length][token_length];
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
	compare_mat<odata_t, sequence_length, token_length>(output, &result_filename, &log_filename);
	std::cout << "test end" << std::endl;
}
