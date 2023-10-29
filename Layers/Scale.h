#pragma once
template<typename T, size_t rows, size_t, size_t cols>
void scale(T A[rows][cols], T result[rows][cols], T scale_factor) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
           result[i][j] = A[i][j] * scale_factor;
        }
    }
}