#pragma once

#include "hls_stream.h"

template<typename T, int rows, int cols>
void replicate2(
	hls::stream<T> input[cols],
	hls::stream<T> result1[cols],
	hls::stream<T> result2[cols]
)
{
	T in;
replicate2_row_loop:
	for (int i = 0; i < rows; i++)
	{
	replicate2_cols_loop:
		for (int j = 0; j < cols; j++)
		{
			input[j].read(in);
			result1[j].write(in);
			result2[j].write(in);
		}

	}
}

template<typename T, int rows, int cols>
void replicate3(
	hls::stream<T> input[cols],
	hls::stream<T> result1[cols],
	hls::stream<T> result2[cols],
	hls::stream<T> result3[cols]
)
{
	T in;
replicate3_row_loop:
	for (int i = 0; i < rows; i++)
	{
	replicate3_cols_loop:
		for (int j = 0; j < cols; j++)
		{
			input[j].read(in);
			result1[j].write(in);
			result2[j].write(in);
			result3[j].write(in);
		}
	}
}

template<typename T, int rows, int cols>
void replicate4(
	hls::stream<T> input[cols],
	hls::stream<T> result1[cols],
	hls::stream<T> result2[cols],
	hls::stream<T> result3[cols],
	hls::stream<T> result4[cols]
)
{
	T in;
replicate4_row_loop:
	for (int i = 0; i < rows; i++)
	{
	replicate4_cols_loop:
		for (int j = 0; j < cols; j++)
		{
			input[j].read(in);
			result1[j].write(in);
			result2[j].write(in);
			result3[j].write(in);
			result4[j].write(in);
		}
	}
}

template<typename T, int rows, int cols, int num>
void replicate(
	hls::stream<T> input[cols],
	hls::stream<T> result[num][cols]
)
{
	T in;
replicate_row_loop:
	for (int i = 0; i < rows; i++)
	{
	replicate_cols_loop:
		for (int j = 0; j < cols; j++)
		{
			input[j].read(in);

			for (int k = 0; k < num; k++)
			{
				result[k][j].write(in);
			}
		}
	}
}

template<typename T, int rows, int cols>
void split3(
	hls::stream<T> input[3][cols],
	hls::stream<T> result1[cols],
	hls::stream<T> result2[cols],
	hls::stream<T> result3[cols]
)
{
	T in1;
	T in2;
	T in3;

split3_row_loop:
	for (int i = 0; i < rows; i++)
	{
	split3_col_loop:
		for (int j = 0; j < cols; j++)
		{

			input[0][j].read(in1);
			input[1][j].read(in2);
			input[2][j].read(in3);
			result1[j].write(in1);
			result2[j].write(in2);
			result3[j].write(in3);
		}
	}
}