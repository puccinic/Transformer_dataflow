#pragma once
#include <functional>
#include <cmath>
#define M_PI 3.14159265358979323846

template<typename T>
T relu(T x) {
	return x > (T)0 ? x : (T) 0;
}

#ifdef GELU
double _gelu(double x) {
	return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * std::pow(x, 3))));
}

template<typename T>
T gelu(T x) {
	return (T)_gelu((double)x);
}
#endif

#ifdef ERF
double _erf(double x) {
	const double a1 = 0.254829592;
	const double a2 = -0.284496736;
	const double a3 = 1.421413741;
	const double a4 = -1.453152027;
	const double a5 = 1.061405429;
	const double p = 0.3275911;

	int sign = (x < 0) ? -1 : 1;
	x = std::abs(x);

	double t = 1.0 / (1.0 + p * x);
	double result = sign * (1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * std::exp(-x * x)));

	return result;
}

template<typename T>
T erf(T x) {
	return (T)_erf((double)x);
}
#endif

template<typename T, int rows, int cols>
void activation(T input[rows][cols], T result[rows][cols]) {
	#pragma HLS ARRAY_PARTITION variable = input dim = 2 complete
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			result[i][j] = relu<T>(input[i][j]);
		}
	}
}
