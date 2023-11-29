#pragma once
template <typename T, size_t rows, size_t cols>
void transpose_matrix(T input[rows][cols], T result[cols][rows]) {
transpose_outer_loop:
    for (size_t i = 0; i < rows; i++) {
    transpose_inner_loop:
        for (size_t j = 0; j < cols; j++) {
            result[j][i] = input[i][j];
        }
    }
}