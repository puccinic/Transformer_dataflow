#pragma once

#include "hls_stream.h"
#include "hls_vector.h"
#include "MatMul.h"
#include "SoftMax.h"

template<int bitWidth, int intWidth, int sequence_length, int token_length>
void scaledotatt(
	hls::stream<hls::vector<ap_fixed<bitWidth, intWidth>, token_length>> &query,
	hls::stream<hls::vector<ap_fixed<bitWidth, intWidth>, token_length>> &key,
	hls::stream<hls::vector<ap_fixed<bitWidth, intWidth>, token_length>> &value,
#ifdef USING_MASKED_SOFTMAX
	hls::stream<hls::vector<bool, sequence_length>> &input_mask,
#endif
	hls::stream<hls::vector<ap_fixed<bitWidth, intWidth>, token_length>> &result
)
{
	hls::stream<hls::vector<ap_fixed<bitWidth, intWidth>, sequence_length>, sequence_length> softmax_att("softmax_att");

	#pragma HLS DATAFLOW
	matmul_scale_softmax<bitWidth, intWidth, bitWidth, intWidth, bitWidth, intWidth, sequence_length,token_length,sequence_length>(
		query,
		key,
		SCALE_FACTOR,
	#ifdef USING_MASKED_SOFTMAX
		input_mask,
	#endif
		softmax_att
	);
	matmul<bitWidth, intWidth, bitWidth, intWidth, bitWidth, intWidth, sequence_length, sequence_length, token_length>(
		softmax_att,
		value,
		result
	);
}
