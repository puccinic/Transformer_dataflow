#pragma once

#include <cmath>
#include<cassert>
#include "hls_stream.h"

template<typename T, int channels, int size>
void batch_norm(
	hls::stream<T> input[size],
	hls::stream<T> gamma[channels],
	hls::stream<T> beta[channels],
    hls::stream<T> mean[channels],
    hls::stream<T> variance[channels],
	hls::stream<T> result[size]
)
{
    T in;
    T g;
    T b;
    T avg;
    T var;
    T batchnorm_rst;
    constexpr T epsilon = std::numeric_limits<T>::epsilon();

batch_norm_loop1:
    for (int i = 0; i < channels; i++)
    {
        gamma[i].read(g);
        beta[i].read(b);
        mean[i].read(avg);
        variance[i].read(var);

    batch_norm_loop2:
        for (int j = 0; j < size; j++)
        {
            /* code */
            input[j].read(in);
            batchnorm_rst = ((((in - avg)/hls::sqrt(var + epsilon)) * g) + b);
            result[j].write(batchnorm_rst);
        }
    }
}