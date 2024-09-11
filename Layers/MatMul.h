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
	hls::stream<aT>& A,
	hls::stream<bT>& B,
	rT scale_factor,
	hls::stream<rT>& result
)
{
	static aT a[rows][hidden]{};
	static bT b[cols][hidden]{};
	rT dot_prod_vec_rst;
	rT dot_prod_rst;

matmul_transpose_scale_load_A_rows_loop:
	for (int i = 0; i < rows; i++)
	{
		#pragma HLS UNROLL
	matmul_transpose_scale_load_A_cols_loop:
		for (int j = 0; j < hidden; j++)
		{
			#pragma HLS UNROLL
			A.read(a[i][j]);
		}
	}
matmul_transpose_scale_load_B_rows_loop:
	for (int i = 0; i < cols; i++)
	{
		#pragma HLS UNROLL
	matmul_transpose_scale_load_B_cols_loop:
		for (int j = 0; j < hidden; j++)
		{
			#pragma HLS UNROLL
			B.read(b[i][j]);
		}
	}
matmul_transpose_scale_compute_row_loop:
	for (int i = 0; i < rows; i++)
	{
		#pragma HLS UNROLL
	matmul_transpose_scale_compute_col_loop:
		for (int j = 0; j < cols; j++)
		{
			#pragma HLS UNROLL
			dot_product<aT, bT, rT, hidden>(a[i], b[j], dot_prod_rst);
			dot_prod_vec_rst = dot_prod_rst / scale_factor;
			result.write(dot_prod_vec_rst);
		}
	}
}

template<typename aT, typename bT, typename rT, int rows, int hidden, int cols>
void matmul_transpose(
	hls::stream<aT>& A,
	hls::stream<bT>& B,
	hls::stream<rT>& result
)
{
	matmul_transpose_scale<aT, bT, rT, rows, hidden, cols>(A, B, 1, result);
}

template<typename T, int rows, int cols>
void transpose(
	hls::stream<T>& A,
	hls::stream<T>& At
)
{
	T transpose_tmp[rows][cols]{};
	T at[rows]{};
transpose_load_A_rows_loop:
	for (int i = 0; i < rows; i++)
	{
		#pragma HLS UNROLL
		for (int j = 0; j < cols; j++)
		{
			#pragma HLS UNROLL
			A.read(transpose_tmp[i][j]);
		}
	}
transpose_loop1:
	for (int i = 0; i < cols; i++)
	{
		#pragma HLS UNROLL
	transpose_loop2:
		for (int j = 0; j < rows; j++)
		{
			At.write(transpose_tmp[j][i]);
		}
	}
}

template<typename aT, typename bT, typename rT, int rows, int hidden, int cols>
void matmul(
	hls::stream<aT>& A,
	hls::stream<bT>& B,
	hls::stream<rT>& result
)
{
	hls::stream<bT, cols*hidden> Bt;
	#pragma HLS DATAFLOW
	transpose<bT, hidden, cols>(B, Bt);
	matmul_transpose<aT, bT, rT, rows, hidden, cols>(A, Bt, result);
}
