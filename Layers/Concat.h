#pragma once 

template<typename T, size_t rows, size_t cols1, size_t cols2, size_t mat_num>
void concat_cols(T matrix1[rows][cols1], T matrix2[rows][cols2], T result[rows][cols1+cols2]) {
    for (int i = 0; i < rows; i++) {
        result[i] = new T[cols1 + cols2];
        for (int j = 0; j < cols1; j++) {
            result[i][j] = matrix1[i][j];
        }
        for (int j = 0; j < cols2; j++) {
            result[i][cols1 + j] = matrix2[i][j];
        }
    }
}

template<typename T, size_t rows1, size_t rows2, size_t cols>
void concat_rows(T matrix1[rows1][cols], T matrix2[rows2][cols], T result[rows1 + rows2][cols]) {
    for (int i = 0; i < rows1; i++) {
        result[i] = new T[cols];
        for (int j = 0; j < cols; j++) {
            result[i][j] = matrix1[i][j];
        }
    }

    for (int i = 0; i < rows2; i++) {
        result[rows1 + i] = new T[cols];
        for (int j = 0; j < cols; j++) {
            result[rows1 + i][j] = matrix2[i][j];
        }
    }
}

