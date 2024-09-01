#pragma once

#include <cmath>
#include<cassert>
#include "hls_stream.h"

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

    // Non-restoring square-root algorithm
fxp_sqrt_loop:
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

template<typename iT, typename gT, typename bT, typename mT, typename vT, typename rT, int channels, int size>
void batch_norm(
	hls::stream<iT> input[size],
	hls::stream<gT> gamma[channels],
	hls::stream<bT> beta[channels],
    hls::stream<mT> mean[channels],
    hls::stream<vT> variance[channels],
	hls::stream<rT> result[size]
)
{
    iT in;
    gT g;
    bT b;
    mT avg;
    vT var;
    rT batchnorm_rst;
    #ifndef USING_FIXED_POINT
        constexpr vT epsilon = std::numeric_limits<vT>::epsilon();
        mT tmpIn;
    #else
        rT std_dev;
        vT epsilon = EPSILON;
    #endif
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
            #ifndef USING_FIXED_POINT
                tmpIn = in;
                batchnorm_rst = ((((tmpIn - avg)/hls::sqrt(var + epsilon)) * g) + b);
            #else
                rT tmp = var + epsilon;
                fxp_sqrt<8, 4, 8, 4>(std_dev, tmp);
                batchnorm_rst = ((((in - avg)/std_dev) * g) + b);
            #endif
            result[j].write(batchnorm_rst);
        }
    }
}

template<typename iT, typename gT, typename bT, typename rT, int channels, int size>
void opt_batch_norm(
	hls::stream<iT> input[size],
	hls::stream<gT> gamma[channels],
	hls::stream<bT> beta[channels],
	hls::stream<rT> result[size]
)
{
    iT in;
    gT g;
    bT b;
    rT batchnorm_rst;
    #ifndef USING_FIXED_POINT
        gT tmpIn;
    #endif

batch_norm_loop1:
    for (int i = 0; i < channels; i++)
    {
        gamma[i].read(g);
        beta[i].read(b);

    batch_norm_loop2:
        for (int j = 0; j < size; j++)
        {
            /* code */
            input[j].read(in);
            #ifndef USING_FIXED_POINT
                tmpIn = in;
                batchnorm_rst = tmpIn*g + b;
            #else
                batchnorm_rst = in*g + b;
            #endif
            result[j].write(batchnorm_rst);
        }
    }
}