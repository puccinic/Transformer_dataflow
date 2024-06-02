#pragma once

#include <cmath>
#include<cassert>
#include "hls_stream.h"
#include "hls_vector.h"

template<typename T, int channels, int size>
void batch_norm(
	hls::stream<hls::vector<T, size>> &input,
	hls::stream<hls::vector<T, channels>> &gamma,
	hls::stream<hls::vector<T, channels>> &beta,
    hls::stream<hls::vector<T, channels>> &mean,
    hls::stream<hls::vector<T, channels>> &variance,
	hls::stream<hls::vector<T, size>> &result
)
{
    hls::vector<T, size> in;
    hls::vector<T, channels> g;
    hls::vector<T, channels> b;
    hls::vector<T, channels> avg;
    hls::vector<T, channels> var;
    hls::vector<T, size> batchnorm_rst;
    constexpr T epsilon = std::numeric_limits<T>::epsilon();

    gamma.read(g);
    beta.read(b);
    mean.read(avg);
    variance.read(var);

batch_norm_loop:
    for (int i = 0; i < channels; i++)
    {
        input.read(in);

        batchnorm_rst = ((((in - avg[i])/hls::sqrt(var[i] + epsilon)) * g[i]) + b[i]);

        result.write(batchnorm_rst);
    }
}