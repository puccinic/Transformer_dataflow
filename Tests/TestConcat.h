#include "TestUtils.h"
#include "Concat.h"

template<typename T, size_t rows, size_t cols>
void test_concat(std::string* matA_filename,
	std::string* matB_filename,
	std::string* result_gold_filename,
	std::string* log_filename) {

	T matrices[2][rows][cols]{};
	load_arr<T, rows*cols>((T*)matrices[0], matA_filename);
	load_arr<T, rows*cols>((T*)matrices[1], matB_filename);

	T result[rows][cols*2]{};

	concat_cols<T, rows, cols, 2>(matrices, result);
	
	compare_mat<T, rows, 2*cols>(result, result_gold_filename, log_filename);
}