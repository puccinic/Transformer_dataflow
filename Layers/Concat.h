#pragma once

#include "hls_stream.h"
#include "hls_vector.h"

template<typename T, int size, int num>
void vector_concat
(
    hls::vector<T, size> &vec_in,
    hls::vector<T, size*num> &vec_res,
    int n
)
{
vector_concat_loop:
    for (int j = 0; j < size; j++)
    {
        vec_res[j + n*size] = vec_in[j];
    }
}

template<typename T, int size, int num>
void vector_concat_list
(
    hls::stream<hls::vector<T, size>> vec_list[num],
    hls::stream<hls::vector<T, size*num>> &result
)
{
    hls::vector<T, size> concat_tmp;
    hls::vector<T, size*num> concat_rst;
concat_list_loop:
    for (int k = 0; k < num; k++)
    {
        vec_list[k].read(concat_tmp);
        vector_concat<T, size, num>(concat_tmp, concat_rst, k);
    }
    result.write(concat_rst);
}

template<typename T, int rows, int cols,  int mat_num>
void concat_cols
(
    hls::stream<hls::vector<T, cols>> matrices[mat_num],
    hls::stream<hls::vector<T, cols*mat_num>> &result
)
{
concat_cols_loop:
    for (int i = 0; i < rows; i++)
    {
        vector_concat_list<T, cols, mat_num>(matrices, result);
    }
}
