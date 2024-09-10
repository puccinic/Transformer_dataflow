#pragma once

#include "hls_stream.h"
#include "MatMul.h"

template <typename T, int size>
void get_max(
	T input[size],
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
	T input[size],
	T result[size]
)
{
	static T max = 0;
	T sum = 0;
	T softmax_tmp[size]{};

	get_max<T, size>(input, max);

softmax_exp_loop:
	for (int i = 0; i < size; i++)
	{
		softmax_tmp[i] = hls::exp(input[i] - max);
		sum += softmax_tmp[i];
	}

softmax_result_loop:
	for (int i = 0; i < size; i++)
	{
		result[i] = softmax_tmp[i] / sum;
	}
}

template<typename T, int size>
void masked_sofmax(
	T input[size],
	T result[size]
)
{
	static T max = 0;
	static int mask = 0;

	T sum = 0;
	T masksoftmax_tmp[size]{};

	get_max<T, size>(input, max);

masked_softmax_exp_loop:
	for (int i = 0; i < size; i++)
	{
		#pragma HLS UNROLL
		if (mask != i)
		{
			masksoftmax_tmp[i] = hls::exp((float) (input[i] - max));
			sum += masksoftmax_tmp[i];
		}
		else
		{
			masksoftmax_tmp[i] = 0;
		}
	}

	mask++;

	if (mask >= size)
	{
		/* restarts value */
		mask = 0;
	}

masked_softmax_result_loop:
	for (int i = 0; i < size; i++)
	{
		if(sum != 0)
		{
			result[i] = masksoftmax_tmp[i] / sum;
		}
		else
		{
			result[i] = 0;
		}
	}
}


template<typename aT, typename bT, typename rT, int rows, int hidden, int cols>
void matmul_scale_masked_softmax(
	hls::stream<aT>& A,
	hls::stream<bT>& B,
	rT scale_factor,
	hls::stream<rT>& result
)
{
	aT a[rows][hidden]{};
	bT b[cols][hidden]{};
	rT matsoftmask_tmp[cols]{};
	rT scaled_dot_prod_vec_rst[cols]{};
	rT scaled_dot_prod_rst;

matmul_transpose_scale_softmask_load_A_rows_loop:
	for (int i = 0; i < rows; i++)
	{
		#pragma HLS UNROLL
	matmul_transpose_scale_softmask_load_A_cols_loop:
		for (int j = 0; j < hidden; j++)
		{
			#pragma HLS UNROLL
			A.read(a[i][j]);
		}

	}

matmul_transpose_scale_softmask_load_B_rows_loop:
	for (int i = 0; i < cols; i++)
	{
		#pragma HLS UNROLL
	matmul_transpose_scale_softmask_load_B_cols_loop:
		for (int j = 0; j < hidden; j++)
		{
			#pragma HLS UNROLL
			B.read(b[i][j]);
		}
	}

matmul_transpose_scale_softmask_compute_row_loop:
	for (int i = 0; i < rows; i++)
	{
		#pragma HLS UNROLL
	matmul_transpose_scale_softmask_compute_col_loop:
		for (int j = 0; j < cols; j++)
		{
			#pragma HLS UNROLL
			dot_product<aT,bT,rT,hidden>(a[i], b[j], scaled_dot_prod_rst);
			matsoftmask_tmp[j] = scaled_dot_prod_rst / scale_factor;
		}

		masked_sofmax<rT,cols>(matsoftmask_tmp, scaled_dot_prod_vec_rst);

	matmul_transpose_scale_softmask_store_col_loop:
		for (int j = 0; j < cols; j++)
		{
			#pragma HLS UNROLL
			result.write(scaled_dot_prod_vec_rst[j]);
		}
	}
}
