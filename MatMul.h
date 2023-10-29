#pragma once
template<typename T, size_t rows, size_t hidden, size_t cols>
void matmul(T A[rows][hidden], T B[hidden][cols], T result[rows][cols]) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i][j] = 0;
            for (size_t k = 0; k < hidden; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}