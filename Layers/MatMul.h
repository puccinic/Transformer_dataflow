#pragma once

#include "hls_stream.h"

template<typename T, int size>
void dot_product(
	T A[size],
	T B[size],
	T &result
)
{
	T dotprod_tmp = 0;
	for (int i = 0; i < size; i++)
	{
		dotprod_tmp += A[i] * B[i];
	}
	result = dotprod_tmp;
}

template<typename T, int rows, int hidden, int cols>
void matmul_transpose_scale(
	hls::stream<T> A[hidden],
	hls::stream<T> B[hidden],
	T scale_factor,
	hls::stream<T> result[cols]
)
{
	T a[rows][hidden]{};
	T b[cols][hidden]{};
	T dot_prod_vec_rst;
	T dot_prod_rst;

matmul_transpose_scale_load_A_rows_loop:
	for (int i = 0; i < rows; i++)
	{
	matmul_transpose_scale_load_A_cols_loop:
		for (int j = 0; j < hidden; j++)
		{
			A[j].read(a[i][j]);
		}
	}

matmul_transpose_scale_load_B_rows_loop:
	for (int i = 0; i < cols; i++)
	{
	matmul_transpose_scale_load_B_cols_loop:
		for (int j = 0; j < hidden; j++)
		{
			B[j].read(b[i][j]);
		}
	}

matmul_transpose_scale_compute_row_loop:
	for (int i = 0; i < rows; i++)
	{
	matmul_transpose_scale_compute_col_loop:
		for (int j = 0; j < cols; j++)
		{
			dot_product<T, hidden>(a[i], b[j], dot_prod_rst);
			dot_prod_vec_rst = dot_prod_rst / scale_factor;
			result[j].write(dot_prod_vec_rst);
		}
	}
}

template<typename T, int rows, int hidden, int cols>
void matmul_transpose(
	hls::stream<T> A[hidden],
	hls::stream<T> B[hidden],
	hls::stream<T> result[cols]
)
{
	matmul_transpose_scale<T, rows, hidden, cols>(A, B, 1, result);
}

template<typename T, int rows, int cols>
void transpose(
	hls::stream<T> A[cols],
	hls::stream<T> At[rows]
)
{
	T transpose_tmp[rows][cols]{};
	T at[rows]{};
transpose_load_A_rows_loop:
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			A[j].read(transpose_tmp[i][j]);
		}
	}
transpose_loop1:
	for (int i = 0; i < cols; i++)
	{
	transpose_loop2:
		for (int j = 0; j < rows; j++)
		{
			At[j].write(transpose_tmp[j][i]);
		}
	}
}

template<typename T, int rows, int hidden, int cols>
void matmul(
	hls::stream<T> A[hidden],
	hls::stream<T> B[cols],
	hls::stream<T> result[cols]
)
{
	hls::stream<T, cols> Bt[hidden]{};
	#pragma HLS DATAFLOW
	transpose<T, hidden, cols>(B, Bt);
	matmul_transpose<T, rows, hidden, cols>(A, Bt, result);
}
