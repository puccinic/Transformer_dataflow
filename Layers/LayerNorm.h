#pragma once

#include <cmath>
#include<cassert>
#include "hls_stream.h"
#include "hls_vector.h"

template <int W2, int IW2, int W1, int IW1>
void fxp_sqrt(
    ap_fixed<W2, IW2>& result,
    ap_fixed<W1, IW1>& in_val
)
{
    enum
    {
        QW = (IW1 + 1) / 2 + (W2 - IW2) + 1
    }; // derive max root width
    enum
    {
        SCALE = (W2 - W1) - (IW2 - (IW1 + 1) / 2)
    }; // scale (shift) to adj initial remainer value
    enum
    {
        ROOT_PREC = QW - (IW1 % 2)
    };
    assert((IW1 + 1) / 2 <= IW2); // Check that output format can accommodate full result

    ap_uint<QW> q = 0;      // partial sqrt
    ap_uint<QW> q_star = 0; // diminished partial sqrt
    ap_int<QW + 2> s; // scaled remainder initialized to extracted input bits

    if (SCALE >= 0)
    {
        s = in_val.range(W1 - 1, 0) << (SCALE);
    }
    else
    {
        s = ((in_val.range(W1 - 1, 0) >> (0 - (SCALE + 1))) + 1) >> 1;
    }

fxp_sqrt_loop:
    // Non-restoring square-root algorithm
    for (int i = 0; i <= ROOT_PREC; i++)
    {
        if (s >= 0)
        {
            s = 2 * s - (((ap_int<QW + 2>(q) << 2) | 1) << (ROOT_PREC - i));
            q_star = q << 1;
            q = (q << 1) | 1;
        }
        else
        {
            s = 2 * s + (((ap_int<QW + 2>(q_star) << 2) | 3) << (ROOT_PREC - i));
            q = (q_star << 1) | 1;
            q_star <<= 1;
        }
    }
    // Round result by "extra iteration" method
    if (s > 0)
    {
        q = q + 1;
    }
    // Truncate excess bit and assign to output format
    result.range(W2 - 1, 0) = ap_uint<W2>(q >> 1);
}


template<typename T, int channels, int size>
void layer_norm(
	hls::stream<hls::vector<T, size>> &input,
	T epsilon,
	hls::stream<hls::vector<T, size>> gamma,
	hls::stream<hls::vector<T, size>> beta,
	hls::stream<hls::vector<T, size>> result
)
{
    hls::vector<T, size> in;
    hls::vector<T, size> g;
    hls::vector<T, size> b;
    hls::vector<T, size> avg_diff;
    hls::vector<T, size> avg_square;
    hls::vector<T, size> tmp1;
    hls::vector<T, size> tmp2;
    hls::vector<T, size> tmp3;
    hls::vector<T, size> tmp4;
    hls::vector<T, size> rst;
    T sum;
    T mean;
    T square_sum;
    T variance;

    gamma.read(g);
    beta.read(b);

layer_norm_outer_loop:
	for (int i = 0; i < channels; i++)
    {
        input.read(in);
		sum = in.reduce_add();
		mean = sum / size;

		avg_diff = in - mean;
		avg_square = avg_diff * avg_diff;
		variance = avg_square.reduce_add() / size;
        T std_dev;
        #if defined(USING_APFIXED)
            fxp_sqrt<IN_WIDTH, IN_IWIDTH, IN_WIDTH, IN_IWIDTH>(std_dev, variance);
        #else
		    std_dev = hls::sqrt(variance);
        #endif /*using ap_fixed */
        tmp1 = in - mean;
        tmp2 = tmp1 * g;
        tmp3 = std_dev + epsilon;
        tmp4 = tmp2 / tmp3;
	    rst = tmp4 + b;

        result.write(rst);
	}
}

template<typename T, int channels, int size>
void batch_norm(
	hls::stream<hls::vector<T, size>> &input,
	T epsilon,
	hls::stream<hls::vector<T, size>> &gamma,
	hls::stream<hls::vector<T, size>> &beta,
    hls::stream<hls::vector<T, size>> &mean,
    hls::stream<hls::vector<T, size>> &stddev,
	hls::stream<hls::vector<T, size>> &result
)
{
    hls::vector<T, size> in;
    hls::vector<T, size> g;
    hls::vector<T, size> b;
    hls::vector<T, size> avg;
    hls::vector<T, size> std_dev;
    hls::vector<T, size> tmp1;
    hls::vector<T, size> tmp2;
    hls::vector<T, size> tmp3;
    hls::vector<T, size> tmp4;
    hls::vector<T, size> rst;

    gamma.read(g);
    beta.read(b);
    mean.read(avg);
    stddev.read(std_dev);

    for (int i = 0; i < channels; i++)
    {
        input.read(in);

        tmp1 = in - avg;
        tmp2 = tmp1 * g;
        tmp3 = std_dev + epsilon;
        tmp4 = tmp2 / tmp3;
	    rst = tmp4 + b;

        result.write(rst);
    }
}