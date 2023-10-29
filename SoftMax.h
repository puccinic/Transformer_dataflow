#pragma once 
#include <cmath>

template<typename T, size_t size>
void softmax(T input[size], T result[size], T scale_factor) {
    T sum = 0;
    T tmp[size] = {};
    for (size_t i = 0; i < size; i++) {
        tmp[i] = std::exp(input[i]);
        sum += std::exp(input[i]);
    }

    for (size_t i = 0; i < size; i++) {
        result[i] =(int) ((tmp[i]/sum) * scale_factor);
    }
}