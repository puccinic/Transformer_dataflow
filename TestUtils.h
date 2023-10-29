#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

template<typename T, size_t size>
void load_arr(T arr[size], std::string* filename) {
    std::ifstream file(*filename);
    for (size_t i = 0; i < size; i++) {
        std::string line;
        std::getline(file, line);
        std::stringstream stream;
        stream << line;
        stream >> arr[i];
    }
}

template<typename T, size_t rows, size_t cols>
void load_mat(T mat[rows][cols], std::string* filename) {
    std::ifstream file(*filename);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::string line;
            std::getline(file, line);
            std::stringstream stream;
            stream << line;
            stream >> mat[i][j];
        }
    }
}

template<typename T, size_t size>
void compare_vec(T vec[size], std::string* vec_filename, std::string* log_filename) {
    std::ifstream file(*vec_filename);
    std::ofstream log(*log_filename);
    bool good_result = true;
    for (size_t i = 0; i < size; i++) {
        std::string line;
        std::getline(file, line);
        std::stringstream stream;
        stream << line;
        T num = 0;
        stream >> num;
        log << vec[i] << " " << num << std::endl;
        if (vec[i] != num) {
            good_result = false;
        }
    }
    if (good_result) {
        std::cout << "Test Passsed!" << std::endl;
        log << "Test Passsed!" << std::endl;
    }
    else {
        std::cout << "Test Failed!" << std::endl;
        log << "Test Failed!" << std::endl;
    }
}

template<typename T, size_t rows, size_t cols>
void compare_mat(T mat[rows][cols], std::string* mat_filename, std::string* log_filename) {
    std::ifstream file(*mat_filename);
    std::ofstream log(*log_filename);
    bool good_result = true;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::string line;
            std::getline(file, line);
            std::stringstream stream;
            stream << line;
            T num = 0;
            stream >> num;
            log << mat[i][j] << " " << num << std::endl;
            if (mat[i][j] != num) {
                good_result = false;
            }
        }
    }
    if (good_result) {
        std::cout << "Test Passsed!" << std::endl;
        log << "Test Passsed!" << std::endl;
    }
    else {
        std::cout << "Test Failed!" << std::endl;
        log << "Test Failed!" << std::endl;
    }
}

