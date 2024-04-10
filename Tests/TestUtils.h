#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <assert.h>

template<class T, int size>
void load_vector(
	hls::vector<T, size> &arr,
	std::ifstream &file
)
{
	T element;
	for (int i = 0; i < size; i++)
	{
		std::string line;
		std::getline(file, line);
		std::stringstream stream;
		stream << line;
		stream >> element;
		arr[i] = element;
	}
}

template<class T, int depth, int size>
void load_stream(
	hls::stream<hls::vector<T, size>> &in_stream,
	std::ifstream &file
)
{
	hls::vector<T, size> vec;
	for (int i = 0; i < depth; i++)
	{
		load_vector<T,size>(vec, file);
		in_stream.write(vec);
	}
}

template<class T, int length, int depth, int size>
void load_stream_array(
	hls::stream<hls::vector<T, size>> *in_stream,
	std::string &filename
)
{
	hls::vector<T, size> vec;
	std::ifstream file(filename);
	for (int i = 0; i < length; i++)
	{
		load_stream<T, depth, size>(in_stream[i], file);
	}
}

template<class T, int depth, int size>
void compare_stream(
	hls::stream<hls::vector<T, size>> &in_stream,
	std::string* vec_filename,
	std::string* log_filename
)
{
	hls::vector<T, size> vec;
	std::string line;
	std::ifstream file(*vec_filename);
	std::ofstream log(*log_filename);
	std::stringstream stream;
	T num = 0;
	double error = 0;
	int mismatch_count = 0;
	double avg_error = 0;
	bool good_result = true;
	for (int j = 0; j < depth; j++)
	{
		in_stream.read(vec);

		for (int i = 0; i < size; i++)
		{
			std::getline(file, line);
			stream << line;
			stream >> num;
			log << vec[i] << " " << num;

			if (vec[i] != num)
			{
				error = 0;

				if (num != 0)
				{
					error = ((double) (((num - vec[i])) / num)) * 100;
				}
				else if (vec[i] != 0)
				{
					error = ((double) (((vec[i]- num)) / vec[i])) * 100;
				}

				log << " -miss relative error: " << error << "%";
				avg_error +=  std::abs(error);
				mismatch_count++;
				good_result = false;
			}

			log << std::endl;
		}
	}

	avg_error = mismatch_count > 0 ? avg_error / mismatch_count : 0;

	if (good_result)
	{
		std::cout << "Test Passsed!" << std::endl;
			log << "Test Succeded with 0 mismatches!" << std::endl;
	}
	else
	{
		std::cout << "Number of mismatchs: " << mismatch_count
			<< " average relative error:  " << avg_error << std::endl;

		log << "Number of mismatchs: " << mismatch_count
			<< " average relative error:  " << avg_error << std::endl;
	}
}