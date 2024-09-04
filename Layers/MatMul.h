#pragma once

#include "hls_stream.h"

template<typename aT, typename bT, typename rT, int size>
void dot_product(
	aT A[size],
	bT B[size],
	rT &result
)
{
	rT dotprod_tmp = 0;
	for (int i = 0; i < size; i++)
	{
		dotprod_tmp += A[i] * B[i];
	}
	result = dotprod_tmp;
}

template<typename aT, typename bT,typename rT, int rows, int hidden, int cols>
void matmul_transpose_scale(
	hls::stream<aT> A[hidden],
	hls::stream<bT> B[hidden],
	rT scale_factor,
	hls::stream<rT> result[cols]
)
{
	aT a[rows][hidden]{};
	bT b[cols][hidden]{};
	rT dot_prod_vec_rst;
	rT dot_prod_rst;

matmul_transpose_scale_load_A_rows_loop:
	for (int i = 0; i < rows; i++)
	{
		#pragma HLS UNROLL factor=rows/64
	matmul_transpose_scale_load_A_cols_loop:
		for (int j = 0; j < hidden; j++)
		{
			#pragma HLS UNROLL factor=hidden/64
			A[j].read(a[i][j]);
		}
	}

matmul_transpose_scale_load_B_rows_loop:
	for (int i = 0; i < cols; i++)
	{
		#pragma HLS UNROLL factor=cols/64
	matmul_transpose_scale_load_B_cols_loop:
		for (int j = 0; j < hidden; j++)
		{
			#pragma HLS UNROLL factor=hidden/64
			B[j].read(b[i][j]);
		}
	}

matmul_transpose_scale_compute_row_loop:
	for (int i = 0; i < rows; i++)
	{
		#pragma HLS UNROLL factor=rows/64
	matmul_transpose_scale_compute_col_loop:
		for (int j = 0; j < cols; j++)
		{
			#pragma HLS UNROLL factor=cols/64
			dot_product<aT, bT, rT, hidden>(a[i], b[j], dot_prod_rst);
			dot_prod_vec_rst = dot_prod_rst / scale_factor;
			result[j].write(dot_prod_vec_rst);
		}
	}
}

template<typename aT, typename bT, typename rT, int rows, int hidden, int cols>
void matmul_transpose(
	hls::stream<aT> A[hidden],
	hls::stream<bT> B[hidden],
	hls::stream<rT> result[cols]
)
{
	matmul_transpose_scale<aT, bT, rT, rows, hidden, cols>(A, B, 1, result);
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
		#pragma HLS UNROLL factor=rows/64
		for (int j = 0; j < cols; j++)
		{
			#pragma HLS UNROLL factor=cols/64
			A[j].read(transpose_tmp[i][j]);
		}
	}
transpose_loop1:
	for (int i = 0; i < cols; i++)
	{
		#pragma HLS UNROLL factor=cols/64
	transpose_loop2:
		for (int j = 0; j < rows; j++)
		{
			At[j].write(transpose_tmp[j][i]);
		}
	}
}

template<typename aT, typename bT, typename rT, int rows, int hidden, int cols>
void matmul(
	hls::stream<aT> A[hidden],
	hls::stream<bT> B[cols],
	hls::stream<rT> result[cols]
)
{
	hls::stream<bT, cols> Bt[hidden]{};
	#pragma HLS DATAFLOW
	transpose<bT, hidden, cols>(B, Bt);
	matmul_transpose<aT, bT, rT, rows, hidden, cols>(A, Bt, result);
}
