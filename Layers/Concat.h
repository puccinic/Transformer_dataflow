#pragma once 

template<typename T, int rows, int cols,  int mat_num>
void concat_cols(T matrices[mat_num][rows][cols], T result[rows][cols*mat_num]) {
concat_cols_row_loop:
    for (int i = 0; i < rows; i++) {
    concat_cols_mat_num_loop:
        for (size_t k = 0; k < mat_num; k++) {
        concat_cols_col_loop:
            for (int j = 0; j < cols; j++) {
                result[i][j +k*cols] = matrices[k][i][j];
            }
        }
    }
}

template<typename T, size_t rows,  size_t cols, size_t mat_num>
void concat_rows(T matrices[mat_num][rows][cols], T result[rows*mat_num][cols]) {
concat_rows_mat_num_loop:
    for (size_t k = 0; k < mat_num; k++) {
    concat_rows_row_loop:
        for (int i = 0; i < rows; i++) {
        concat_rows_col_loop:
            for (int j = 0; j < cols; j++) {
                result[i + k * rows][j] = matrices[k][i][j];
            }
        }
    }
}

