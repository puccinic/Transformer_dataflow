#pragma once 

template<typename T, size_t rows, size_t cols,  size_t mat_num>
void concat_cols(T (*matrices[mat_num])[rows][cols], T result[rows][cols*mat_num]) {
    for (int i = 0; i < rows; i++) {
        for (size_t k = 0; k < mat_num; k++) {
            for (int j = 0; j < cols; j++) {
                result[i][j +k*cols] = (*matrices[k])[i][j];
            }
        }
    }
}

template<typename T, size_t rows,  size_t cols, size_t mat_num>
void concat_rows(T (*matrices[mat_num])[rows][cols], T result[rows*mat_num][cols]) {
    for (size_t k = 0; k < mat_num; k++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i + k * rows][j] = (*matrices[k])[i][j];
            }
        }
    }
}

