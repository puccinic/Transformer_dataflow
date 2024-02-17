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
	T tmp = 0;

get_max_loop:
	for (int i = 1; i < size; i++)
	{
		if (tmp < input[i])
		{
      	tmp = input[i];
    	}
  	}
	*result = tmp;
}

template<typename T, int size>
void softmax(
	hls::vector<T, size> &input,
	hls::vector<T, size> &result
)
{
	T max;
	T sum = 0;
	hls::vector<T, size> tmp;

	get_max<T, size>(input, max);

softmax_exp_loop:
	for (int i = 0; i < size; i++)
	{
		tmp[i] = hls::exp(input[i] - max);
	}
	sum = tmp.reduce_add();

softmax_result_loop:
	for (int i = 0; i < size; i++)
	{
		result[i] = tmp[i] / sum;
	}
}

template<typename T, int size>
void masked_sofmax(
	hls::vector<T, size> &input,
	hls::vector<T, size> &mask,
	hls::vector<T, size> &result
)
{
	T max = 0;
	T sum = 0;
	hls::vector<T, size> tmp;

	get_max<T, size>(input, max);

masked_softmax_exp_loop:
	for (int i = 0; i < size; i++)
	{
		tmp[i] = mask[i] ? (T) hls::exp((double) (input[i] - max)) : (T) 0;
	}
	sum = tmp.reduce_add();

masked_softmax_result_loop:
	for (int i = 0; i < size; i++)
	{
		if(tmp[i] != 0 && sum != 0)
		{
			result[i] = tmp[i] / sum;
		}
		else
		{
			result[i] = 0;
		}
	}
}


template<typename T, int rows, int hidden, int cols>
void matmul_scale_masked_softmax
(
	hls::stream<hls::vector<T, hidden>> &A,
	hls::stream<hls::vector<T, hidden>> &B,
	T scale_factor,
	hls::stream<hls::vector<T, cols>> &input_mask,
	hls::stream<hls::vector<T, cols>> &result
)
{
	hls::vector<T, hidden> a;
	hls::vector<T, hidden> b;
	hls::vector<T, cols> tmp;
	hls::vector<T, cols> mask;
	hls::vector<T, cols> rst;
	T dot_prod_rst;

matmul_transpose_scale_row_loop:
	for (int i = 0; i < rows; i++)
	{
		A.read(a);
		input_mask.read(mask);

	matmul_transpose_scale_col_loop:
		for (int j = 0; j < cols; j++)
		{
			B.read(b);
			dot_product<T,hidden>(a, b, dot_prod_rst);
			tmp[j] = dot_prod_rst / scale_factor;
		}
		masked_sofmax<T,cols>(tmp, mask, rst);
		result.write(rst);
	}
}