#pragma once
template<typename T, size_t size>
void vecadd(T A[size], T B[size], T result[size]) {
vecadd_loop:
	for (size_t i = 0; i < size; i++) {
		result[i] = A[i] + B[i];
	}
}