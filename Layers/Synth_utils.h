#pragma once

#include "hls_stream.h"
#include "hls_vector.h"

template<typename T, int rows, int cols>
void replicate2
(
	hls::stream<hls::vector<T, cols>> &input,
	hls::stream<hls::vector<T, cols>> &result1,
	hls::stream<hls::vector<T, cols>> &result2
)
{
	hls::vector<T, cols> in;
	for (int i = 0; i < rows; i++)
	{
		input.read(in);
		result1.write(in);
		result2.write(in);
	}
}

template<typename T, int rows, int cols>
void replicate3
(
	hls::stream<hls::vector<T, cols>> &input,
	hls::stream<hls::vector<T, cols>> &result1,
	hls::stream<hls::vector<T, cols>> &result2,
	hls::stream<hls::vector<T, cols>> &result3
)
{
	hls::vector<T, cols> in;
	for (int i = 0; i < rows; i++)
	{
		input.read(in);
		result1.write(in);
		result2.write(in);
		result3.write(in);
	}
}

template<typename T, int rows, int cols>
void replicate4
(
	hls::stream<hls::vector<T, cols>> &input,
	hls::stream<hls::vector<T, cols>> &result1,
	hls::stream<hls::vector<T, cols>> &result2,
	hls::stream<hls::vector<T, cols>> &result3,
	hls::stream<hls::vector<T, cols>> &result4
)
{
	hls::vector<T, cols> in;
	for (int i = 0; i < rows; i++)
	{
		input.read(in);
		result1.write(in);
		result2.write(in);
		result3.write(in);
		result4.write(in);
	}
}

template<typename T, int rows, int cols>
void split3
(
	hls::stream<hls::vector<T, cols>> input[3],
	hls::stream<hls::vector<T, cols>> &result1,
	hls::stream<hls::vector<T, cols>> &result2,
	hls::stream<hls::vector<T, cols>> &result3
)
{
	hls::vector<T, cols> in1;
	hls::vector<T, cols> in2;
	hls::vector<T, cols> in3;
	for (int i = 0; i < rows; i++)
	{
		input[0].read(in1);
		input[1].read(in2);
		input[3].read(in3);
		result1.write(in1);
		result2.write(in2);
		result3.write(in3);
	}
}