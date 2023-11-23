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


template<typename T, size_t size>
void compare_vec(T vec[size], std::string* vec_filename, std::string* log_filename) {
	std::ifstream file(*vec_filename);
	std::ofstream log(*log_filename);
	int mismatch_count = 0;
	bool good_result = true;
	for (size_t i = 0; i < size; i++) {
		std::string line;
		std::getline(file, line);
		std::stringstream stream;
		stream << line;
		T num = 0;
		stream >> num;
		log << vec[i] << " " << num;
		if (vec[i] != num) {
			log << " -miss";
			mismatch_count++;
			good_result = false;
		}
		log << std::endl;
	}
	if (good_result) {
		std::cout << "Test Passsed!" << std::endl;
		log << "Test Succeded with 0 mismatches!" << std::endl;
	}
	else {
		std::cout << "Number of mismatchs: " << mismatch_count << std::endl;
		log << "Number of mismatchs: " << mismatch_count << std::endl;
	}
}

template<typename T, size_t rows, size_t cols>
void compare_mat(T mat[rows][cols], std::string* mat_filename, std::string* log_filename) {
	std::ifstream file(*mat_filename);
	std::ofstream log(*log_filename);
	int mismatch_count = 0;
	bool good_result = true;
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			std::string line;
			std::getline(file, line);
			std::stringstream stream;
			stream << line;
			T num = 0;
			stream >> num;
			log << mat[i][j] << " " << num;
			if (mat[i][j] != num) {
				log << " -miss";
				mismatch_count++;
				good_result = false;
			}
			log << std::endl;
		}
	}
	if (good_result) {
		std::cout << "Test Passsed!" << std::endl;
		log << "Test Succeded with 0 mismatches!" << std::endl;
	}
	else {
		std::cout << "Number of mismatchs: " << mismatch_count << std::endl;
		log << "Number of mismatchs: " << mismatch_count << std::endl;
	}
}

