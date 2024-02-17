#pragma once

#include "hls_stream.h"
#include "hls_vector.h"

template<typename T, int size>
void dot_product(
	hls::vector<T, cols> &A,
	hls::vector<T, cols> &B,
	T &result
)
{
	hls::vector<T, cols> tmp = A * B;
	result = tmp.reduce_add();
}

template<typename T, int rows, int hidden, int cols>
void matmul_transpose(
	hls::stream<hls::vector<T, hidden>> &A,
	hls::stream<hls::vector<T, hidden>> &B,
	hls::stream<hls::vector<T, cols>> &result
)
{
	hls::vector<T, hidden> a;
	hls::vector<T, hidden> b;
	hls::vector<T, cols> rst;
	T dot_prod_rst;

matmul_row_loop:
	for (int i = 0; i < rows; i++)
	{
		A.read(a);

	matmul_result_loop:
		for (int j = 0; j < cols; i++)
		{
			B.read(b);
			dot_product<T,hidden>(a, b, dot_prod_rst);
			rst[j] = rst;
		}
		result.write(rst);
	}
}

template<typename T, int rows, int hidden, int cols>
void matmul_transpose_scale(
	hls::stream<hls::vector<T, hidden>> &A,
	hls::stream<hls::vector<T, hidden>> &B,
	T scale_factor,
	hls::stream<hls::vector<T, cols>> &result
)
{
	hls::vector<T, hidden> a;
	hls::vector<T, hidden> b;
	hls::vector<T, cols> rst;
	T dot_prod_rst;

matmul_transpose_scale_row_loop:
	for (int i = 0; i < rows; i++)
	{
		A.read(a);

	matmul_transpose_scale_col_loop:
		for (int j = 0; j < cols; j++)
		 {
			B.read(b);
			dot_product<T,hidden>(a, b, dot_prod_rst);
			rst[j] = dot_prod_rst / scale_factor;
		}
		result.write(rst);
	}
}