#pragma once
#include "TestUtils.h"
#include "MatMul.h"

template<typename T, size_t rows, size_t hidden, size_t cols>
void test_matmul(std::string* matA_filename,
    std::string* matB_filename,
    std::string* matResGold_filename,
    std::string* log_filename) {

    T A[rows][hidden]{};
    load_mat<T, rows, hidden>(A, matA_filename);

    T B[hidden][cols]{};
    load_mat<T, hidden, cols>(B, matB_filename);

    T result[rows][cols]{};

    matmul<T, rows, hidden, cols>(A, B, result);

    compare_mat<T, rows, cols>(result, matResGold_filename, log_filename);
}
