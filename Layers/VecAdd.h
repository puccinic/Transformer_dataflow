#pragma once
template<typename T, int size>
void vecadd(T A[size], T B[size], T result[size]) {
vecadd_loop:
	for (int i = 0; i < size; i++) {
		#pragma HLS PIPELINE
		result[i] = A[i] + B[i];
	}
}
