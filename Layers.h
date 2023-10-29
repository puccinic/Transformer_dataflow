#pragma once

template<typename T, size_t size>
void add(T A[size], T B[size], T result[size]);

template<typename T, size_t rows, size_t, size_t cols>
void scale(T A[rows][cols], T result[rows][cols], T scale_factor);

template<typename T, size_t rows, size_t cols>
void mat_add(T A[rows][cols], T B[rows][cols], T result[rows][cols]);

template<typename T, size_t rows, size_t hidden, size_t cols>
void matmul(T A[rows][hidden], T B[hidden][cols], T result[rows][cols]);

template <typename T, size_t rows, size_t cols>
void transpose_matrix(T input[rows][cols], T result[rows][cols]);

template<int rows, int hidden, int cols>
void linear(int input[rows][hidden], int weights[hidden][cols], int biases[cols], int result[rows][cols]);

template<typename T, size_t size>
void relu(T input[size], T result[size]);

template<typename T, size_t size>
void layer_normalization(T input[size], T ressult[size],
    double epsilon = 1e-5, double gamma = 1.0, double beta = 0.0);

template<typename T, size_t size>
void softmax(T input[size], T result[size], T scale_factor);
