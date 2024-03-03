#pragma once

#include "hls_stream.h"
#include "hls_vector.h"

template<typename T, int size>
void dot_product
(
	hls::vector<T, size> &A,
	hls::vector<T, size> &B,
	T &result
)
{
	hls::vector<T, size> dotprod_tmp = A * B;
	result = dotprod_tmp.reduce_add();
}

template<typename T, int rows, int hidden, int cols>
void matmul_transpose_scale
(
	hls::stream<hls::vector<T, hidden>> &A,
	hls::stream<hls::vector<T, hidden>> &B,
	T scale_factor,
	hls::stream<hls::vector<T, cols>> &result
)
{
	hls::vector<T, hidden> a;
	hls::vector<T, hidden> b;
	hls::vector<T, cols> dot_prod_vec_rst;
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
			dot_prod_vec_rst[j] = dot_prod_rst / scale_factor;
		}
		result.write(dot_prod_vec_rst);
	}
}

template<typename T, int rows, int hidden, int cols>
void matmul_transpose
(
	hls::stream<hls::vector<T, hidden>> &A,
	hls::stream<hls::vector<T, hidden>> &B,
	hls::stream<hls::vector<T, cols>> &result
)
{
	matmul_transpose_scale<T, rows, hidden, cols>(A, B, 1, result);
}

template<typename T, int rows, int cols>
void transpose
(
	hls::stream<hls::vector<T, cols>> &A,
	hls::stream<hls::vector<T, rows>> &At
)
{
	hls::vector<T, cols> transpose_tmp[rows];
	hls::vector<T, rows> at;
	for (int i = 0; i < rows; i++)
	{
		A.read(transpose_tmp[i]);
	}

	for (int i = 0; i < cols; i++)
	{
		for (int j = 0; j < rows; j++)
		{
			at[j] = transpose_tmp[j][i];
		}
		At.write(at);
	}
}

template<typename T, int rows, int hidden, int cols>
void matmul
(
	hls::stream<hls::vector<T, hidden>> &A,
	hls::stream<hls::vector<T, cols>> &B,
	hls::stream<hls::vector<T, cols>> &result
)
{
	hls::stream<hls::vector<T, hidden>> Bt;

	#pragma HLS DATAFLOW
	transpose<T, hidden, cols>(B, Bt);
	matmul_transpose<T, rows, hidden, cols>(A, Bt, result);
}
