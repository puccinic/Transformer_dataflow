#pragma once
#include "hls_stream.h"

template<typename T, int size, int num>
void vector_concat(
    T vec_in[size],
    T vec_res[size*num],
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
void vector_concat_list(
    hls::stream<T> vec_list[num][size],
    hls::stream<T> result[size*num]
)
{
    T concat_tmp[size]{};
    T concat_rst[size*num]{};
concat_list_loop:
    for (int k = 0; k < num; k++)
    {
    concat_list_load_loop:
        for (int i = 0; i < size; i++)
        {
            vec_list[k][i].read(concat_tmp[i]);
        }

        vector_concat<T, size, num>(concat_tmp, concat_rst, k);
    }

concat_list_store_loop:
    for (int k = 0; k < size*num; k++)
    {
        result[k].write(concat_rst[k]);
    }

}

template<typename T, int rows, int cols,  int mat_num>
void concat_cols(
    hls::stream<T> matrices[mat_num][cols],
    hls::stream<T> result[cols*mat_num]
)
{
concat_cols_loop:
    for (int i = 0; i < rows; i++)
    {
        vector_concat_list<T, cols, mat_num>(matrices, result);
    }
}
