#pragma once
#include "TestUtils.h"
#include "VecAdd.h"
template<typename T, size_t size>
void test_vecadd(std::string* vec1_filename,
	std::string* vec2_filename,
	std::string* vecResGold_filename,
	std::string* log_filename) {

	T vec1[size]{};
	load_arr<T, size>(vec1, vec1_filename);

	T vec2[size]{};
	load_arr<T, size>(vec2, vec2_filename);

	T vecres[size]{};

	vecadd<T, size>(vec1, vec2, vecres);

	compare_vec<T, size>(vecres, vecResGold_filename, log_filename);
}
