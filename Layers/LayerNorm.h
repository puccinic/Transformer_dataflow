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


template<
    int bitWidthI,
    int intWidthI,
    int bitWidthG,
    int intWidthG,
    int bitWidthB,
    int intWidthB,
    int bitWidthR,
    int intWidthR,
    int channels,
    int size
>
void layer_norm(
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, size>> &input,
	hls::stream<hls::vector<ap_fixed<bitWidthG, intWidthG>, size>> &gamma,
	hls::stream<hls::vector<ap_fixed<bitWidthB, intWidthB>, size>> &beta,
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, size>> &result
)
{
    hls::vector<ap_fixed<bitWidthI, intWidthI>, size> in;
    hls::vector<ap_fixed<bitWidthG, intWidthG>, size> g;
    hls::vector<ap_fixed<bitWidthB, intWidthB>, size> b;
    hls::vector<ap_fixed<bitWidthR, intWidthR>, size> avg_diff;
    hls::vector<ap_fixed<bitWidthR, intWidthR>, size> avg_square;
    hls::vector<ap_fixed<bitWidthR, intWidthR>, size> layernorm_rst;
    hls::vector<ap_fixed<bitWidthR, intWidthR>, size> layernorm_tmp1;
    hls::vector<ap_fixed<bitWidthR, intWidthR>, size> layernorm_tmp2;
    hls::vector<ap_fixed<bitWidthR, intWidthR>, size> layernorm_tmp3;
    hls::vector<ap_fixed<bitWidthR, intWidthR>, size> layernorm_tmp4;
    ap_fixed<bitWidthR, intWidthR> sum = 0;
    static ap_fixed<bitWidthR, intWidthR> mean = 0;
    ap_fixed<bitWidthR, intWidthR> square_sum = 0;
    static ap_fixed<bitWidthR, intWidthR> variance = 0;
    static ap_fixed<bitWidthR, intWidthR> std_dev = 0;
    static ap_fixed<bitWidthR, intWidthR> epsilon = 1 >> (bitWidthR - intWidthR);
    gamma.read(g);
    beta.read(b);

layer_norm_outer_loop:
	for (int i = 0; i < channels; i++)
    {
        input.read(in);

        //compute mean
		sum = in.reduce_add();
		mean = sum / size;

        //compute variance
		avg_diff = in - mean;
		avg_square = avg_diff * avg_diff;
		variance = avg_square.reduce_add() / size;
        fxp_sqrt<bitWidthR, intWidthR, bitWidthR, intWidthR>(std_dev, variance);
        layernorm_tmp1 = in - mean;
        layernorm_tmp2 = layernorm_tmp1 * g;
        layernorm_tmp3 = std_dev + epsilon;
        layernorm_tmp4 = layernorm_tmp2 / layernorm_tmp3;
	    layernorm_rst = layernorm_tmp4 + b;


        result.write(layernorm_rst);
	}
}

template<
    int bitWidthI,
    int intWidthI,
    int bitWidthG,
    int intWidthG,
    int bitWidthB,
    int intWidthB,
    int bitWidthM,
    int intWidthM,
    int bitWidthS,
    int intWidthS,
    int bitWidthR,
    int intWidthR,
    int channels,
    int size
>
void batch_norm(
	hls::stream<hls::vector<ap_fixed<bitWidthI, intWidthI>, size>> &input,
	hls::stream<hls::vector<ap_fixed<bitWidthG, intWidthG>, size>> &gamma,
	hls::stream<hls::vector<ap_fixed<bitWidthB, intWidthB>, size>> &beta,
    hls::stream<hls::vector<ap_fixed<bitWidthM, intWidthM>, size>> &mean,
    hls::stream<hls::vector<ap_fixed<bitWidthS, intWidthS>, size>> &stddev,
	hls::stream<hls::vector<ap_fixed<bitWidthR, intWidthR>, size>> &result
)
{
    hls::vector<ap_fixed<bitWidthI, intWidthI>, size> in;
    hls::vector<ap_fixed<bitWidthG, intWidthG>, size> g;
    hls::vector<ap_fixed<bitWidthB, intWidthB>, size> b;
    hls::vector<ap_fixed<bitWidthM, intWidthM>, size> avg;
    hls::vector<ap_fixed<bitWidthS, intWidthS>, size> std_dev;
    hls::vector<ap_fixed<bitWidthR, intWidthR>, size> batchnorm_tmp1;
    hls::vector<ap_fixed<bitWidthR, intWidthR>, size> batchnorm_tmp2;
    hls::vector<ap_fixed<bitWidthR, intWidthR>, size> batchnorm_tmp3;
    hls::vector<ap_fixed<bitWidthR, intWidthR>, size> batchnorm_tmp4;
    hls::vector<ap_fixed<bitWidthR, intWidthR>, size> batchnorm_rst;
    static ap_fixed<bitWidthR, intWidthR> epsilon = 1 >> (bitWidthR - intWidthR);

    gamma.read(g);
    beta.read(b);
    mean.read(avg);
    stddev.read(std_dev);
batch_norm_loop:
    for (int i = 0; i < channels; i++)
    {
        input.read(in);

        batchnorm_tmp1 = in - avg;
        batchnorm_tmp2 = batchnorm_tmp1 * g;
        batchnorm_tmp3 = std_dev + epsilon;
        batchnorm_tmp4 = batchnorm_tmp2 / batchnorm_tmp3;
	    batchnorm_rst = batchnorm_tmp4 + b;

        result.write(batchnorm_rst);
    }
}