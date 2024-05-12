#pragma once

#include "hls_stream.h"
#include "hls_vector.h"
#include "MatMul.h"

template <typename T, int size>
void get_max(
	hls::vector<T, size> &input,
	T &result
)
{
	T max = 0;

get_max_loop:
	for (int i = 1; i < size; i++)
	{
		if (max < input[i])
		{
      	max = input[i];
    	}
  	}
	result = max;
}

template<typename T, int size>
void softmax(
	hls::vector<T, size> &input,
	hls::vector<T, size> &result
)
{
	T max;
	T sum = 0;
	hls::vector<T, size> softmax_tmp;

	get_max<T, size>(input, max);

softmax_exp_loop:
	for (int i = 0; i < size; i++)
	{
		softmax_tmp[i] = hls::exp(input[i] - max);
	}
	sum = softmax_tmp.reduce_add();

softmax_result_loop:
	for (int i = 0; i < size; i++)
	{
		result[i] = softmax_tmp[i] / sum;
	}
}

template<typename T, int size>
void masked_sofmax(
	hls::vector<T, size> &input,
	hls::vector<bool, size> &mask,
	hls::vector<T, size> &result
)
{
	T max = 0;
	T sum = 0;
	hls::vector<T, size> masksoftmax_tmp;

	get_max<T, size>(input, max);

masked_softmax_exp_loop:
	for (int i = 0; i < size; i++)
	{
		if (mask[i])
		{
			masksoftmax_tmp[i] = hls::exp((double) (input[i] - max));
		}
		else
		{
			masksoftmax_tmp[i] = 0;
		}
	}
	sum = masksoftmax_tmp.reduce_add();

masked_softmax_result_loop:
	for (int i = 0; i < size; i++)
	{
		if(masksoftmax_tmp[i] != 0 && sum != 0)
		{
			result[i] = masksoftmax_tmp[i] / sum;
		}
		else
		{
			result[i] = 0;
		}
	}
}


template<
	int bitWidthA,
	int intWidthA,
	int bitWidthB,
	int intWidthB,
	int bitWidthR,
	int intWidthR,
	int rows,
	int hidden,
	int cols
>
void matmul_scale_softmax(
	hls::stream<hls::vector<ap_fixed<bitWidthA, intWidthA>, hidden>> &A,
	hls::stream<hls::vector<ap_fixed<bitWidthB, intWidthB>, hidden>> &B,
	ap_fixed<bitWidthR, intWidthR> scale_factor,
	#ifdef USING_MASKED_SOFTMAX
		hls::stream<hls::vector<bool, cols>> &input_mask,
	#endif
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, cols>> &result
)
{
	hls::vector<ap_fixed<bitWidthA, intWidthA>, hidden> a[rows]{};
	hls::vector<ap_fixed<bitWidthB, intWidthB>, hidden> b[cols]{};
	hls::vector<ap_fixed<bitWidthR, intWidthR>, cols> matsoftmask_tmp;
	#ifdef USING_MASKED_SOFTMAX
		hls::vector<bool, cols> mask[rows]{};
	#endif
	hls::vector<ap_fixed<bitWidthR, intWidthR>, cols> scaled_dot_prod_vec_rst;
	ap_fixed<bitWidthR, intWidthR> scaled_dot_prod_rst;

matmul_transpose_scale_softmask_load_A_loop:
	for (int i = 0; i < rows; i++)
	{
		A.read(a[i]);
	#ifdef USING_MASKED_SOFTMAX
		input_mask.read(mask[i]);
	#endif
	}

matmul_transpose_scale_softmask_load_B_loop:
	for (int j = 0; j < cols; j++)
	{
		B.read(b[j]);
	}

matmul_transpose_scale_softmask_compute_row_loop:
	for (int i = 0; i < rows; i++)
	{
	matmul_transpose_scale_softmask_compute_col_loop:
		for (int j = 0; j < cols; j++)
		{
			dot_product<bitWidthA, intWidthA, bitWidthB, intWidthB, bitWidthR, intWidthR, hidden>(
				a[i],
				b[j],
				scaled_dot_prod_rst
			);
			matsoftmask_tmp[j] = scaled_dot_prod_rst / scale_factor;
		}
		#ifdef USING_MASKED_SOFTMAX
			masked_sofmax<ap_fixed<bitWidthR, intWidthR>, cols>(
				matsoftmask_tmp,
				mask[i],
				scaled_dot_prod_vec_rst
			);
		#else
			softmax<ap_fixed<bitWidthR, intWidthR>, cols>(
				matsoftmask_tmp,
				scaled_dot_prod_vec_rst
			);
		#endif
		result.write(scaled_dot_prod_vec_rst);
	}
}
