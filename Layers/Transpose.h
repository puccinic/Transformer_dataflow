#pragma once
template <typename T, size_t rows, size_t cols>
void transpose_matrix(T input[rows][cols], T result[cols][rows]) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[j][i] = input[i][j];
        }
    }
}