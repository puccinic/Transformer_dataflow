#pragma once
#include <cmath>
#include<cassert>

template <int W2, int IW2, int W1, int IW1>
void fxp_sqrt(ap_fixed<W2, IW2>& result, ap_fixed<W1, IW1>& in_val) {
    enum { QW = (IW1 + 1) / 2 + (W2 - IW2) + 1 }; // derive max root width
    enum {
        SCALE = (W2 - W1) - (IW2 - (IW1 + 1) / 2)
    }; // scale (shift) to adj initial remainer value
    enum { ROOT_PREC = QW - (IW1 % 2) };
    assert((IW1 + 1) / 2 <=
           IW2); // Check that output format can accommodate full result

    ap_uint<QW> q = 0;      // partial sqrt
    ap_uint<QW> q_star = 0; // diminished partial sqrt
    ap_int<QW + 2> s; // scaled remainder initialized to extracted input bits
    if (SCALE >= 0)
        s = in_val.range(W1 - 1, 0) << (SCALE);
    else
        s = ((in_val.range(W1 - 1, 0) >> (0 - (SCALE + 1))) + 1) >> 1;
fxp_sqrt_loop:
    // Non-restoring square-root algorithm
    for (int i = 0; i <= ROOT_PREC; i++) {
        if (s >= 0) {
            s = 2 * s - (((ap_int<QW + 2>(q) << 2) | 1) << (ROOT_PREC - i));
            q_star = q << 1;
            q = (q << 1) | 1;
        } else {
            s = 2 * s +
                (((ap_int<QW + 2>(q_star) << 2) | 3) << (ROOT_PREC - i));
            q = (q_star << 1) | 1;
            q_star <<= 1;
        }
    }
    // Round result by "extra iteration" method
    if (s > 0)
        q = q + 1;
    // Truncate excess bit and assign to output format
    result.range(W2 - 1, 0) = ap_uint<W2>(q >> 1);
}


template<typename T, int channels, int size>
void layer_norm(
	T input[channels][size], 
	T epsilon, 
	T gamma[size], 
	T beta[size], 
	T result[channels][size]
) {
layer_norm_outer_loop:
    #pragma HLS ARRAY_PARTITION variable = input dim = 2 complete
	for (int i = 0; i < channels; i++) {
		#pragma HLS UNROLL
		T sum = 0.0;
	layer_norm_avg_loop:
		for (int j = 0; j < size; j++) {
			sum += (T) input[i][j];
		}
		T mean = sum / size;
		T variance = 0.0;
	layer_norm_variance_loop:
		for (int j = 0; j < size; j++) {
			T tmp = (((T) input[i][j]) - mean);
			variance += tmp * tmp;
		}
		variance = variance / size;
        T std_dev;
		//std_dev = hls::sqrt(variance);
        fxp_sqrt<IN_WIDTH, IN_IWIDTH, IN_WIDTH, IN_IWIDTH>(std_dev, variance);
	layer_norm_result_loop:
		for (int j = 0; j < size; j++) {
			result[i][j] = (((input[i][j] - mean) * gamma[j]) / std_dev) + beta[j];
		}
	}
}
