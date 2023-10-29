#pragma once
template<typename T, size_t size>
void relu(T input[size], T result[size]) {
	for (size_t i = 0; i < size; i++) {
		result[size] = input[i] > 0 ? input[i] : 0;
	}
}