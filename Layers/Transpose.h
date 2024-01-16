#pragma once
template <typename T, int rows, int cols>
void transpose_matrix(T input[rows][cols], T result[cols][rows]) {
    //#pragma HLS ARRAY_PARTITION variable = input dim = 2 complete        
transpose_outer_loop:
    for (int i = 0; i < rows; i++) {
//#pragma HLS PIPELINE
    transpose_inner_loop:
        for (int j = 0; j < cols; j++) {
//#pragma HLS UNROLL
            result[j][i] = input[i][j];
        }
    }
}
