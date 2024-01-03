#pragma once
#include <ap_fixed.h>

template <int src_width, int src_i_width, int res_width, int res_i_width, int size>
void ap_fixed_resize_vec(
    ap_fixed<src_width, src_i_width> src[size], 
    ap_fixed<res_width, res_i_width> res[size]
) {
    constexpr int low_range = src_width - 1 - res_width;
    for (int i = 0; i < size; i++) {
        res[i] = src[i].range(src_width - 1, low_range);
    }
}