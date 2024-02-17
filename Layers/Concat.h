#pragma once

#include "hls_stream.h"
#include "hls_vector.h"

template<typename T, int rows, int cols,  int mat_num>
void concat_cols
(
    hls::stream<hls::vector<T, cols>> matrices[mat_num],
    hls::stream<hls::vector<T, cols*mat_num>> result
)
{

    hls::vector<T, cols> tmp;
    hls::vector<T, cols*mat_num> rst;
concat_cols_row_loop:
    for (int i = 0; i < rows; i++)
    {
    concat_cols_mat_num_loop:
        for (int k = 0; k < mat_num; i++)
        {
            matrices[k].read(tmp);
        concat_cols_col_loop:
            for (int j = 0; j < cols; j++)
            {
                rst[j +k*cols] = tmp[j];
            }
        }
        result.write(rst);
    }
}
