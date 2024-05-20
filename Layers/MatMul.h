#pragma once

#include "hls_stream.h"
#include "hls_vector.h"
#include <ap_fixed.h>

template<
	int bitWidthA,
	int intWidthA,
	int bitWidthB,
	int intWidthB,
	int bitWidthR,
	int intWidthR,
	int size
>
void dot_product(
	hls::vector<ap_fixed<bitWidthA, intWidthA>, size> &A,
	hls::vector<ap_fixed<bitWidthB, intWidthB>, size> &B,
	ap_fixed<bitWidthR, intWidthR> &result
)
{
	hls::vector<ap_fixed<bitWidthR, intWidthR>, size> dotprod_tmp;
dot_prod_loop:
	for (int i = 0; i < size; i++)
	{
		dotprod_tmp[i] = A[i] * B[i];
	}
	result = dotprod_tmp.reduce_add();
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
void matmul_transpose_scale(
	hls::stream<hls::vector<ap_fixed<bitWidthA, intWidthA>, hidden>> &A,
	hls::stream<hls::vector<ap_fixed<bitWidthB, intWidthB>, hidden>> &B,
	ap_fixed<bitWidthR, intWidthR> scale_factor,
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, cols>> &result
)
{
	hls::vector<ap_fixed<bitWidthA, intWidthA>, hidden> a[rows]{};
	hls::vector<ap_fixed<bitWidthB, intWidthB>, hidden> b[cols]{};
	hls::vector<ap_fixed<bitWidthR, intWidthR>, cols> dot_prod_vec_rst;
	ap_fixed<bitWidthR, intWidthR> dot_prod_rst = 0;

matmul_transpose_scale_load_A_loop:
	for (int i = 0; i < rows; i++)
	{
		A.read(a[i]);
	}

matmul_transpose_scale_load_B_loop:
	for (int j = 0; j < cols; j++)
	{
		B.read(b[j]);
	}

matmul_transpose_scale_compute_row_loop:
	for (int i = 0; i < rows; i++)
	{
	matmul_transpose_scale_compute_col_loop:
		for (int j = 0; j < cols; j++)
		{
			dot_product<bitWidthA, intWidthA, bitWidthB, intWidthB, bitWidthR, intWidthR, hidden>(
				a[i],
				b[j],
				dot_prod_rst
			);
			dot_prod_vec_rst[j] = dot_prod_rst / scale_factor;
		}
		result.write(dot_prod_vec_rst);
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
void matmul_transpose(
	hls::stream<hls::vector<ap_fixed<bitWidthA, intWidthA>, hidden>> &A,
	hls::stream<hls::vector<ap_fixed<bitWidthB, intWidthB>, hidden>> &B,
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, cols>> &result
)
{
	matmul_transpose_scale<bitWidthA, intWidthA, bitWidthB, intWidthB, bitWidthR, intWidthR, rows, hidden, cols>(
		A,
		B,
		1,
		result
	);
}

template<typename T, int rows, int cols>
void transpose(
	hls::stream<hls::vector<T, cols>> &A,
	hls::stream<hls::vector<T, rows>> &At
)
{
	hls::vector<T, cols> transpose_tmp[rows]{};
	hls::vector<T, rows> at;
transpose_loopA:
	for (int i = 0; i < rows; i++)
	{
		A.read(transpose_tmp[i]);
	}
transpose_loop1:
	for (int i = 0; i < cols; i++)
	{
	transpose_loop2:
		for (int j = 0; j < rows; j++)
		{
			at[j] = transpose_tmp[j][i];
		}
		At.write(at);
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
void matmul(
	hls::stream<hls::vector<ap_fixed<bitWidthA, intWidthA>, hidden>> &A,
	hls::stream<hls::vector<ap_fixed<bitWidthB, intWidthB>, cols>> &B,
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, cols>> &result
)
{
	hls::stream<hls::vector<ap_fixed<bitWidthB, intWidthB>, hidden>, cols> Bt;
	#pragma HLS DATAFLOW
	transpose<ap_fixed<bitWidthB, intWidthB>, hidden, cols>(B, Bt);
	matmul_transpose<bitWidthA, intWidthA, bitWidthB, intWidthB, bitWidthR, intWidthR, rows, hidden, cols>(
		A,
		Bt,
		result
	);
}
