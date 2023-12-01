#pragma once
template <typename T, int rows, int cols>
void transpose_matrix(T input[rows][cols], T result[cols][rows]) {
transpose_outer_loop:
    for (int i = 0; i < rows; i++) {
#pragma HLS UNROLL
    transpose_inner_loop:
        for (int j = 0; j < cols; j++) {
#pragma HLS UNROLL
            result[j][i] = input[i][j];
        }
    }
}
