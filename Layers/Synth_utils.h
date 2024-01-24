#pragma once

template<typename T, int size>
void replicate2(T input[size], T result1[size], T result2[size]) {
	for (int i = 0; i < size; i++) {
		result1[i] = input[i];
		result2[i] = input[i];
	}
}

template<typename T, int size>
void replicate3(T input[size], T result1[size], T result2[size], T result3[size]) {
	for (int i = 0; i < size; i++) {
		result1[i] = input[i];
		result2[i] = input[i];
		result3[i] = input[i];
	}
}

template<typename T, int size>
void replicate4(T input[size], T result1[size], T result2[size], T result3[size], T result4[size]) {
	for (int i = 0; i < size; i++) {
		result1[i] = input[i];
		result2[i] = input[i];
		result3[i] = input[i];
		result4[i] = input[i];
	}
}

template<typename T, int size>
void split3(T input[3][size], T result1[size], T result2[size], T result3[size]) {
	for (int i = 0; i < size; i++) {
		result1[i] = input[0][i];
		result2[i] = input[1][i];
		result3[i] = input[2][i];
	}
}