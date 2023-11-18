#pragma once
#include "TestUtils.h"
#include "ScaledDotAtt.h"


template<typename T, size_t sequence_length, size_t token_length, T scale_factor>
void test_scaledotatt(std::string* matIn_filename,
	std::string* matMask_filename,
	std::string* result_gold_filename,
	std::string* log_filename) {

	T matIn[sequence_length][token_length]{};
	load_arr<T, sequence_length*token_length>((T*)matIn, matIn_filename);
	T matMask[sequence_length][sequence_length]{};
	load_arr<T, sequence_length*sequence_length>((T*)matMask, matMask_filename);

	T matResult[sequence_length][token_length]{};
	scaledotatt<T, sequence_length, token_length, scale_factor>(matIn, matIn, matIn, matMask, matResult);

	compare_mat<T, sequence_length, token_length>(matResult, result_gold_filename, log_filename);
}