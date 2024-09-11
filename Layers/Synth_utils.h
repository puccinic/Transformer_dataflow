#pragma once

#include "hls_stream.h"

template<typename T, int rows, int cols>
void replicate2(
	hls::stream<T>& input,
	hls::stream<T>& result1,
	hls::stream<T>& result2
)
{
	T in;
replicate2_row_loop:
	for (int i = 0; i < rows; i++)
	{
	replicate2_cols_loop:
		for (int j = 0; j < cols; j++)
		{
			input.read(in);
			result1.write(in);
			result2.write(in);
		}

	}
}

template<typename T, int rows, int cols>
void replicate3(
	hls::stream<T>& input,
	hls::stream<T>& result1,
	hls::stream<T>& result2,
	hls::stream<T>& result3
)
{
	T in;
replicate3_row_loop:
	for (int i = 0; i < rows; i++)
	{
	replicate3_cols_loop:
		for (int j = 0; j < cols; j++)
		{
			input.read(in);
			result1.write(in);
			result2.write(in);
			result3.write(in);
		}
	}
}

template<typename T, int rows, int cols>
void replicate4(
	hls::stream<T>& input,
	hls::stream<T>& result1,
	hls::stream<T>& result2,
	hls::stream<T>& result3,
	hls::stream<T>& result4
)
{
	T in;
replicate4_row_loop:
	for (int i = 0; i < rows; i++)
	{
	replicate4_cols_loop:
		for (int j = 0; j < cols; j++)
		{
			input.read(in);
			result1.write(in);
			result2.write(in);
			result3.write(in);
			result4.write(in);
		}
	}
}

template<typename T, int rows, int cols, int num>
void replicate(
	hls::stream<T>& input,
	hls::stream<T> result[num]
)
{
	T in;
replicate_row_loop:
	for (int i = 0; i < rows; i++)
	{
	replicate_cols_loop:
		for (int j = 0; j < cols; j++)
		{
			input.read(in);

			for (int k = 0; k < num; k++)
			{
				result[k].write(in);
			}
		}
	}
}

template<typename T, int rows, int cols>
void split3(
	hls::stream<T>& input,
	hls::stream<T>& result1,
	hls::stream<T>& result2,
	hls::stream<T>& result3
)
{
	T in1;
	T in2;
	T in3;

	for (int i = 0; i < rows; i++)
	{
		for (int i = 0; i < cols; i++)
		{
			input.read(in1);
			result1.write(in1);
		}

		for (int i = 0; i < cols; i++)
		{
			input.read(in2);
			result2.write(in2);
		}

		for (int i = 0; i < cols; i++)
		{
			input.read(in3);
			result3.write(in3);
		}
	}
}